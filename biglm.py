import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding, SelfAttentionMask
from transformer import AttLSTMLayer
from activations import *


class BIGLM(nn.Module):
    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers, approx, mode=''):
        super(BIGLM, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=local_rank)
        
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.vocab.size)

        # used in response keyword exploration
        self.reduce = nn.Linear(2*embed_dim, embed_dim)
        # used in response keyword exploration
        #self.interpolate = nn.Linear(embed_dim, 1)

        if mode == 1 or mode == 3:
            # only the model with KI component will use gate
            self.gate_q = nn.Linear(embed_dim, embed_dim, bias=True)
            self.gate_x = nn.Linear(embed_dim, embed_dim, bias=True)
        
        self.attn_mask = SelfAttentionMask(device=local_rank)
        
        self.dropout = dropout
        self.device = local_rank

        if approx == "none":
            self.approx = None
        elif approx == "adaptive":
            self.approx = nn.AdaptiveLogSoftmaxWithLoss(self.embed_dim, self.vocab.size, [10000, 20000, 200000])
        else:
            raise NotImplementedError("%s has not been implemented"%approx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.normal_(self.out_proj.weight, std=0.02)
    
    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        #print(cost)
        return torch.mean(cost)

    def mse_loss(self, y_pred, y, mask, avg=True):
        # calculate the mean squared error loss
        cost = ((y_pred - y) * mask) ** 2
        if avg:
            cost = torch.mean(cost, dim=1)
        else:
            cost = torch.sum(cost, dim=1)
        return torch.mean(cost)

    def bow_loss(self, y_pred, y, avg=True):
        """
        bag-of-word loss
        :param y_pred: (batch_size, vocab_size)
        :param y: (batch_size, vocab_size)
        :param avg:
        :return:
        """
        cost = -torch.log(y_pred) * y
        if avg:
            # avg over the sequence
            cost = torch.sum(cost, 1) / (torch.sum(y, 1) + 1e-4)
        else:
            cost = torch.sum(cost, 1)
        # average over cost
        return torch.mean(cost)

    def work(self, inp, args, src_lens, apply_softmax=True):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #print("vocab.padding_idx:", self.vocab.padding_idx)
        padding_mask = torch.eq(inp, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        # added by lixin
        max_src_len = max(src_lens)
        min_src_len = min(src_lens)
        max_tgt_len = seq_len - min_src_len
        msk_src = torch.zeros(bsz, max_src_len)
        msk_tgt = torch.zeros(max_tgt_len, bsz)
        for i in range(bsz):
            src_len_i = src_lens[i]
            tgt_len_i = seq_len - src_len_i
            msk_src[i, :src_len_i] = torch.ones(src_len_i).float()
            msk_tgt[max_tgt_len - tgt_len_i:, i] = torch.ones(tgt_len_i)
        msk_src = msk_src.float().cuda(args.gpu)
        msk_tgt = msk_tgt.float().cuda(args.gpu)  # shape: (max_tgt_len, bsz)
        msk_tgt = msk_tgt.unsqueeze(2)  # shape: (max_tgt_len, bsz, 1)
        padding_mask_src = torch.eq(msk_src, 0).t_()
        # added by lixin
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)
            if args.use_src_attn:
                # x_source and x_target may have some overlap
                x_source = x[:max_src_len]  # (max_src_len, bsz, hsz)
                x_target = x[min_src_len:]  # (max_tgt_len, bsz, hsz)
                x_target, attn, _ = layer(x=x_target, kv=x_source,
                                          self_padding_mask=padding_mask_src, self_attn_mask=None, need_weights=True)
                x = torch.cat([x[:min_src_len], msk_tgt*x_target+(1.0-msk_tgt)*x[min_src_len:]], dim=0)

        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        if args.use_resp_kw:
            query_reprs = []
            for (bid, hid) in enumerate(src_lens):
                query_reprs.append(x[hid, bid])
            query_reprs = torch.stack(query_reprs, 0)
            # one more linear layer with tanh activation
            query_reprs = torch.tanh(self.query_proj(query_reprs))
            query_reprs = query_reprs.repeat(x.size(0), 1, 1)
            gt = torch.sigmoid(self.gate_x(x)+self.gate_q(query_reprs))
            x = gt * x + (1.0 - gt) * query_reprs
            if apply_softmax:
                probs = torch.softmax(self.out_proj(x), -1)
            else:
                probs = self.out_proj(x)
        else:
            if apply_softmax:
                probs = torch.softmax(self.out_proj(x), -1)
            else:
                probs = self.out_proj(x)

        _, pred_y = probs.max(-1)
        
        return probs, pred_y
    
    def forward(self, inp, truth, src_attn_truth, kw_resp_truth, msk, msk_src, src_lens, args, local_rank):
        """

        :param inp: input word
        :param truth: ground truth
        :param src_attn_truth: source attention ground truth
        :param kw_resp_truth: response keyword ground truth
        :param msk: mask for calculating the nll_loss
        :param msk_src: mask for the source input
        :param src_lens: query lengths
        :return:
        """
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        # shape: (seq_len, bsz, dim)
        x = self.tok_embed(inp) + self.pos_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if msk_src is not None:
            #assert args.use_src_attn and self.training
            padding_mask_src = torch.eq(msk_src, 0).t_()
        if not padding_mask.any():
            padding_mask = None

        # added by lixin
        # store the source attentions from multiple transformer layers
        if src_lens is not None:
            #assert args.use_src_attn and self.training
            source_attentions = []
            max_src_len = max(src_lens)
            min_src_len = min(src_lens)
            max_tgt_len = seq_len - min_src_len
            msk_tgt = torch.zeros(max_tgt_len, bsz)
            for i in range(bsz):
                src_len_i = src_lens[i]
                tgt_len_i = seq_len - src_len_i
                msk_tgt[max_tgt_len-tgt_len_i:, i] = torch.ones(tgt_len_i)
            #print(msk_tgt)
            msk_tgt = msk_tgt.float().cuda(local_rank)   # shape: (max_tgt_len, bsz)
            msk_tgt = msk_tgt.unsqueeze(2)
        for layer in self.layers:
            # kv is none, perform self-attention
            x, _, _ = layer(x=x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask, need_weights=False)
            #x_self_attn, _, _ = layer(x=x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask, need_weights=False)

            # kv is not none, perform encoder-decoder attention
            # attn: (tgt_len, bsz, src_len), from the first head
            # x_target: (tgt_len, bsz, hsz)
            # added by lixin
            #x = x_self_attn
            if args.use_src_attn:
                x_source = x[:max_src_len]   # (max_src_len, bsz, hsz)
                x_target = x[min_src_len:]   # (max_tgt_len, bsz, hsz)
                x_target, attn, _ = layer(x=x_target, kv=x_source,
                                          self_padding_mask=padding_mask_src, self_attn_mask=None, need_weights=True)
                x_target = x_target * msk_tgt
                #x[min_src_len:] = msk_tgt * x_target + (1.0 - msk_tgt) * x_self_attn[min_src_len:]
                x = torch.cat([x[:min_src_len], msk_tgt*x_target+(1.0-msk_tgt)*x[min_src_len:]], dim=0)
                # set the source attention value of the non-target words as 0
                attn = attn * msk_tgt  # mask the non-target words
                source_attentions.append(attn)


        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        if args.use_resp_kw:
            #ratio = torch.sigmoid(self.interpolate(x))
            query_reprs = []
            for bid, hid in enumerate(src_lens):
                # regard the representation of token <bos> as query/source representation
                query_reprs.append(x[hid, bid])
            # (bsz, vocab_size)
            query_reprs = torch.stack(query_reprs, dim=0)
            # one more linear layer with tanh activation
            query_reprs = torch.tanh(self.query_proj(query_reprs))
            # size: (seq_len, bsz, embed_dim)
            pred_kw = torch.softmax(self.out_proj(query_reprs), -1)
            # tile the query representation to perform the linear combination
            query_reprs = query_reprs.repeat(x.size(0), 1, 1)
            gt = torch.sigmoid(self.gate_x(x) + self.gate_q(query_reprs))
            x = gt * x + (1.0 - gt) * query_reprs
            pred = torch.softmax(self.out_proj(x), -1)
            nll_loss = self.nll_loss(pred, truth, msk)
            if self.training:
                kw_loss = self.bow_loss(y_pred=pred_kw, y=kw_resp_truth, avg=True)
                loss = nll_loss + kw_loss * 0.2
            else:
                loss = nll_loss
        else:
            # for the model0
            pred = torch.softmax(self.out_proj(x), -1)
            nll_loss = self.nll_loss(pred, truth, msk)
            loss = nll_loss
        # added by lixin
        if args.use_src_attn and self.training:
            # shape: (bsz, src_len)
            # perform max pooling over the attention weight
            pred_src_attn = torch.max(source_attentions[-1], dim=0)[0]
            # add constraint on the source attention
            attn_loss = self.mse_loss(y_pred=pred_src_attn, y=src_attn_truth, mask=msk_src, avg=True)
            loss += attn_loss

        # added by lixin
        _, pred_y = pred.max(-1)
        #msk = msk.cuda(local_rank)
        tot_tokens = msk.float().sum().item()
        acc = torch.eq(pred_y, truth).float().sum().item()
        
        return (pred_y, truth), loss, acc, tot_tokens, bsz


class BIGLM_LSTM(nn.Module):
    # implementation of self-attentative LSTM
    def __init__(self, local_rank, vocab, embed_dim, num_heads, dropout, layers):
        super(BIGLM_LSTM, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.tok_embed = Embedding(num_embeddings=self.vocab.size, embedding_dim=self.embed_dim,
                                   padding_idx=self.vocab.padding_idx)
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(AttLSTMLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout))
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, self.vocab.size)

        self.attn_mask = SelfAttentionMask(device=local_rank)
        self.dropout = dropout
        self.device = local_rank

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost)

    def work(self, inp, args, src_lens, apply_softmax=True):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(inp, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        if apply_softmax:
            probs = torch.softmax(self.out_proj(x), -1)
        else:
            probs = self.out_proj(x)
        _, pred_y = probs.max(-1)
        return probs, pred_y

    def forward(self, inp, truth, msk):
        seq_len, bsz = inp.size()
        self_attn_mask = self.attn_mask(seq_len)
        x = self.tok_embed(inp)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.vocab.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask, self_attn_mask=self_attn_mask)
        x = self.one_more_layer_norm(gelu(self.one_more(x)))
        pred = torch.softmax(self.out_proj(x), -1)

        loss = self.nll_loss(pred, truth, msk)

        _, pred_y = pred.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = torch.eq(pred_y, truth).float().sum().item()

        return (pred_y, truth), loss, acc, tot_tokens, bsz
