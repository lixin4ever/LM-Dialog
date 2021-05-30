import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy
import argparse
import os

from biglm import BIGLM, BIGLM_LSTM
from data import Vocab, DataLoader, s2t, EOS


#gpu = 0


def init_model(m_path, device, vocab):
    ckpt = torch.load(m_path, map_location='cpu')
    # the parameter settings in the saved model
    lm_args = ckpt['args']

    mode2string = {
        0: "basic transformer, use nothing",
        1: "use response keyword",
        2: "use source attention",
        3: "use response keyword + source attention",
        4: "lstm language model with self attention"
    }
    if lm_args.use_src_attn and lm_args.use_resp_kw:
        mode = 3
    elif lm_args.use_src_attn and not lm_args.use_resp_kw:
        mode = 2
    elif not lm_args.use_src_attn and lm_args.use_resp_kw:
        mode = 1
    else:
        mode = 0
    #lm_args.dropout = 0.001
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[], model_path=m_path)
    if lm_args.lm_type == 'transformer':
        lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads,
                         lm_args.dropout, lm_args.layers, lm_args.approx, mode=mode)
    elif lm_args.lm_type == 'lstm':
        mode = 4
        lm_model = BIGLM_LSTM(local_rank=device, vocab=lm_vocab, embed_dim=lm_args.embed_dim,
                              num_heads=lm_args.num_heads, dropout=lm_args.dropout, layers=lm_args.layers)
    else:
        raise Exception("Invalid type %s of language model!!!" % lm_args.lm_type)
    print(lm_model)
    model_description = "Model: %s..." % mode2string[mode]
    print(model_description)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    return lm_model, lm_vocab, lm_args


def greedy(s, lm_model, lm_vocab, lm_args, src_lens):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    dec_words = []
    for l in range(50):
        _, pred = lm_model.work(x, lm_args, src_lens)
        next_tk = []
        for i in range(len(s)):
            next_tk.append(lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))
        s = [sent + [t] for sent, t in zip(s, next_tk)]

        x, m = s2t(s, lm_vocab)
        x = x.cuda(gpu)

    for sent in s:
        if "<eos>" in sent:
            sent = sent[0:sent.index("<eos>")]
        sent = ' '.join(sent)+'\n'
        dec_words.append(sent)
        #print(sent)
    return dec_words


def topk_sampling_decode(x, lm_model, lm_vocab, lm_args, src_lens):
    temperature = 1.0
    k = lm_args.k
    x = x.cuda(gpu)
    ys = x.unsqueeze(1)
    dec_words = []
    filter_value = -float('Inf')
    for step in range(50):
        y_pred, _ = lm_model.work(ys, lm_args, src_lens, apply_softmax=False)
        logit = y_pred[-1, :, :].squeeze() / temperature
        #threshold = min(torch.topk(logit, int(k), dim=-1))
        topk_logit, _ = torch.topk(logit, k, sorted=False, dim=-1)
        threshold = torch.min(topk_logit)
        filtered_logit = torch.where(
            logit < threshold,
            torch.ones_like(logit) * filter_value,
            logit
        )
        prob = torch.softmax(filtered_logit, dim=-1)
        next_y = torch.multinomial(prob, num_samples=1)
        next_tk = lm_vocab.idx2token(next_y.item())
        if next_tk == '<eos>':
            # stop when it comes to <eos> token
            break
        dec_words.append(next_tk)
        ys = torch.cat([ys, next_y.view(1, 1)], dim=0)
    return dec_words


def topk_sampling(s, lm_model, lm_vocab, lm_args, src_lens):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    dec_words = []
    for i in range(len(s)):
        dec_words_i = topk_sampling_decode(x[:len(s[i]), i], lm_model, lm_vocab, lm_args, [src_lens[i]])
        dec_words.append(' '.join(dec_words_i)+'\n')
    return dec_words


def scheduled_topk_sampling_decode(x, lm_model, lm_vocab, lm_args, src_lens):
    temperature = 1.0
    k = lm_args.k
    x = x.cuda(gpu)
    ys = x.unsqueeze(1)
    dec_words = []
    filter_value = -float('Inf')
    p = 1.0
    for step in range(50):
        y_pred, max_pred = lm_model.work(ys, lm_args, src_lens, apply_softmax=False)

        m = torch.distributions.Bernoulli(torch.tensor([p]))
        flag = m.sample().item()
        if flag:
            # use greedy sampling
            prob = torch.softmax(y_pred, dim=-1)
            _, next_y = prob.max(-1)
            next_tk = lm_vocab.idx2token(next_y.item())
        else:
            # use topk sampling
            logit = y_pred[-1, :, :].squeeze() / temperature
            # threshold = min(torch.topk(logit, int(k), dim=-1))
            topk_logit, _ = torch.topk(logit, k, sorted=False, dim=-1)
            threshold = torch.min(topk_logit)
            filtered_logit = torch.where(
                logit < threshold,
                torch.ones_like(logit) * filter_value,
                logit
            )
            prob = torch.softmax(filtered_logit, dim=-1)
            # print(prob.shape)
            # print(prob)
            next_y = torch.multinomial(prob, num_samples=1)
            # print(next_y)
            next_tk = lm_vocab.idx2token(next_y.item())
        if next_tk == '<eos>':
            # stop when it comes to <eos> token
            break
        # next_y and next_tk are two outputs produced in each step.
        dec_words.append(next_tk)
        ys = torch.cat([ys, next_y.view(1, 1)], dim=0)
        if (step + 1) % 5 == 0:
            # halve the probability every 5 steps
            p /= 2.0
    return dec_words


def scheduled_topk_sampling(s, lm_model, lm_vocab, lm_args, src_lens):
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    dec_words = []
    for i in range(len(s)):
        dec_words_i = scheduled_topk_sampling_decode(x[:len(s[i])], lm_model, lm_vocab, lm_args, [src_lens[i]])
        dec_words.append(dec_words_i)
    return dec_words


def topk_greedy_decode(s, x, lm_model, lm_vocab, lm_args, src_lens):
    # NOT used in the following code
    k = lm_args.k
    # shape: (seq_len, )
    x = x.cuda(gpu)
    # shape: (seq_len, 1), batch_first=False
    ys = x.unsqueeze(1)
    #k = 5
    dec_words = []
    for step in range(50):
        y_pred, _ = lm_model.work(ys, lm_args, src_lens)
        # shape: (1, 1, vocab_size)-->(vocab_size,)
        logit = y_pred[-1, :, :].squeeze()
        if k == 0:
            truncated_logit = logit
        else:
            # shape: (k, ), (k, )
            truncated_logit, topk_indices = torch.topk(logit, k, sorted=False, dim=-1)
        next_y = torch.multinomial(truncated_logit, num_samples=1)
        next_y = topk_indices[next_y[0]]
        next_tk = lm_vocab.idx2token(next_y.item())
        if next_tk == '<eos>':
            # stop when it comes to <eos> token
            break
        dec_words.append(next_tk)
        ys = torch.cat([ys, next_y.view(1, 1)], dim=0)
    return dec_words


def topk_greedy(s, lm_model, lm_vocab, lm_args, src_lens):
    # NOT used in the following code
    x, m = s2t(s, lm_vocab)
    x = x.cuda(gpu)
    dec_words = []
    for i in range(len(s)):
        # decoding sentence by sentence
        dec_words_i = topk_greedy_decode(s[i], x[:len(s[i]), i], lm_model, lm_vocab, lm_args, [src_lens[i]])
        dec_words.append(' '.join(dec_words_i)+'\n')
    return dec_words


def beam_decode(s, x, lm_model, lm_vocab, lm_args, src_lens):
    beam_size = lm_args.k
    samples = []
    sample_scores = np.zeros(beam_size)

    num_dead = 0

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).cuda(gpu)

    x = x.cuda(gpu)
    # shape: (seq_len, 1)
    ys = x.unsqueeze(1)

    for step in range(50):
        y_pred, _ = lm_model.work(ys, lm_args, src_lens)
       
        '''
        if step == 0:
            o_len = ys.size(0)
            for i in range(o_len-1):
                last_scores += torch.log(y_pred[i, 0, ys[i, 0]])
        '''

        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_y_scores.flatten()
        # shape: (seq_len, num_live)
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        ys_now = []
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            ys_now.append(copy.copy(ys[:,j]))
        # shape: (seq_len, beam_size - num_dead)
        ys = torch.stack(ys_now, dim = 1) 

        num_live = 0
        last_traces = []
        last_scores = []
        #print("traces_now[i][-1]:", traces_now[i][-1])
        live_idxs = []
        for i in range(len(traces_now)):
            #print("traces_now[i]", traces_now[i])
            if traces_now[i][-1].item() == lm_vocab.token2idx(EOS) or i >= 50:
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                num_live += 1
                live_idxs.append(i)
        live_idxs = torch.LongTensor(live_idxs).cuda(gpu)
        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).cuda(gpu)
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            next_y.append(eid)
        next_y = np.array(next_y).reshape((1, num_live))
        # shape: (1, seq_len)
        next_y = torch.LongTensor(next_y).cuda(gpu)
        # obtain the input at the next decoding step
        ys = ys.index_select(dim=1, index=live_idxs)
        
        ys = torch.cat([ys, next_y], dim=0)
        src_lens = [src_lens[0]]
        src_lens *= num_live

        assert num_live + num_dead == beam_size
        
        # end for loop

    if num_live > 0:
        for i in range(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[i] = last_scores[i]
            num_dead += 1

    for i in range(len(sample_scores)):
        sent_len = float(len(samples[i]))
        lp = np.power(5 + sent_len, 0.9) / np.power(5 + 1, 0.9)
        #sample_scores[i] /= lp
    
    idx_sorted_scores = np.argsort(sample_scores) # ascending order

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) > 0:
            # only reserve the non-empty sentence
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    dec_words = []
    for sample in sorted_samples[::-1]:
        for e in sample:
            e = int(e)
            dec_words.append(lm_vocab.idx2token(e))
        # only print the sentence with the highest score
        # print(''.join(s+dec_words))
        break
    return dec_words


def beam_search(s, lm_model, lm_vocab, lm_args, src_lens):
    x, m = s2t(s, lm_vocab)
    dec_words = []
    for i in range(len(s)):
        dec_words_i = beam_decode(s[i], x[:len(s[i]), i], lm_model, lm_vocab, lm_args, [src_lens[i]])
        dec_words.append(' '.join(dec_words_i)+'\n')
    return dec_words
        #break



if __name__ == '__main__':
    #s = ["马可装备末世", "诸葛亮对线嬴政", "干将莫邪", "复活甲", "貂蝉"]
    #s = ["俞栋", "史树明", "刘晓江", "李丕绩", "闭玮", "王琰", "李华阳", "王龙跃", "王星", "涂兆鹏", "刘乐茂", "黄国平", "韩家龙", "李菁", "张海松"]
    #s = ["机器学习", "腾讯", "ai lab"]
    #s = ["你才神经病", "这道题怎么解", "关于姐弟恋,大家有神马想说的", "停车坐爱枫林晚", "我觉得男人做饭的时候特别性感", "屌丝终有逆袭日"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--decode_type', type=str)
    parser.add_argument('--k', type=int)
    args = parser.parse_args()
    lm_model, lm_vocab, lm_args = init_model('./ckpt/%s' % args.model, args.gpu, args.vocab)
    #print("vocab:", lm_vocab._token2idx)
    lm_model.eval()
    test_path = args.test_data
    k = args.k
    lm_args.k = args.k
    lm_args.gpu = args.gpu
    queries, gold_resps = [], []
    #gpu = args.gpu
    global gpu
    gpu = args.gpu
    print("Model: %s..." % args.model)
    print("Decoding type: %s..." % args.decode_type)
    print("Test data: %s..." % args.test_data)
    with open(test_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            query, resp = line.strip().split('\t')
            # character-based model
            queries.append(''.join(query.split(' ')))
            resp_str = ''.join(resp.split(' '))
            gold_resps.append(' '.join([w for w in resp_str])+'\n')
    print(gold_resps[:10])
    if not os.path.exists('./data/gold_resp.txt'):
        with open('./data/gold_resp.txt', 'w+', encoding='UTF-8') as fp:
            fp.writelines(gold_resps)
    t = []
    src_lens = []
    for sent in queries:
        sent = [w for w in sent] + ["<bos>"]
        # source length will not include the symbol <bos>
        src_lens.append(len(sent) - 1)
        t.append(sent)
    queries = t
    # when decoding type is beam search, args.k is beam size
    # when decoding type is topk decoding, args.k is the sampling size
    #if args.decode_type == 'tg':
    #    pred_resps = topk_greedy(queries, lm_model, lm_vocab, lm_args)
    #elif args.decode_type == 'tf':
    if args.decode_type == 'tk':
        pred_resps = topk_sampling(queries, lm_model, lm_vocab, lm_args, src_lens)
    elif args.decode_type == 'tks':
        # scheduled top-k sampling
        pred_resps = scheduled_topk_sampling(queries, lm_model, lm_vocab, lm_args, src_lens)
    elif args.decode_type == 'bm':
        pred_resps = beam_search(queries, lm_model, lm_vocab, lm_args, src_lens)
    elif args.decode_type == 'gd':
        pred_resps = greedy(queries, lm_model, lm_vocab, lm_args, src_lens)
    else:
        raise ValueError("Invalid decode type %s..." % args.decode_type)
    if 'online' in args.test_data:
        pred_file = './dec_res/pred_resp_online_%s_%s%s.txt' % (args.model, args.decode_type, args.k)
    else:
        pred_file = './dec_res/pred_resp_%s_%s%s.txt' % (args.model, args.decode_type, args.k)
    with open(pred_file, 'w+', encoding='UTF-8') as fp:
        fp.writelines(pred_resps)

    total_lines = []
    for (q, g, p) in zip(queries, gold_resps, pred_resps):
        line = '%s\t%s\t%s\n' % (''.join(q[:-1]), ''.join(g.strip().split(' ')), ''.join(p.strip().split(' ')))
        total_lines.append(line)
    if not os.path.exists('./dec_res'):
        os.mkdir('./dec_res')
    if 'online' in args.test_data:
        total_file = './dec_res/out_%s_%s%s_online.txt' % (args.model, args.decode_type, args.k)
    else:
        total_file = './dec_res/out_%s_%s%s.txt' % (args.model, args.decode_type, args.k)
    with open(total_file, 'w+', encoding='UTF-8') as fp:
        fp.writelines(total_lines)
