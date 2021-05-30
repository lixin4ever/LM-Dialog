import random
import torch
import numpy as np
random.seed(3789)

PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
BUFSIZE = 4096000


def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len -len(x))
        else:
            y = x + [0]*(max_len -len(x))
        ys.append(y)
    return ys


def reformat(src_kws):
    # max_len <==> maximum query length
    max_len = max(len(x) for x in src_kws)
    kws_padded = []
    for kw in src_kws:
        kw_padded = []
        for w in kw:
            if w == '<unk>':
                kw_padded.append(0.0)
            else:
                kw_padded.append(1.0)
        kw_padded += [0.0] * (max_len - len(kw))
        assert len(kw_padded) == max_len
        kws_padded.append(kw_padded)
    return kws_padded


def _back_to_text_for_check(x, vocab):
    w = x.t().tolist()
    for sent in vocab.idx2token(w):
        print (' '.join(sent))


def batchify_original(data, vocab):
    truth, inp, msk, src_lens = [], [], [], []
    src_attn_truth = []
    bsz = len(data)
    for bid, (x, query_kw) in enumerate(data):
        inp.append(x[:-1])
        truth.append(x[1:])
        bos_idx = x.index('<bos>')
        msk.append([1 for i in range(len(x) -1)])
        src_attn_truth.append(query_kw)
        src_lens.append(bos_idx)
    max_src_len = max(src_lens)
    msk_src = torch.zeros(bsz, max_src_len)
    for i in range(bsz):
        msk_src[i, :src_lens[i]] = torch.ones(src_lens[i]).float()

    truth = torch.LongTensor(ListsToTensor(truth, vocab)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    src_attn_truth = torch.FloatTensor(reformat(src_attn_truth)).contiguous()
    return inp, truth, src_attn_truth, msk, msk_src, src_lens


def batchify(data, vocab):
    truth, inp, src_attn_truth, msk, src_lens = [], [], [], [], []
    bsz = len(data)
    kw_resp_truth = torch.zeros(bsz, vocab.size)
    token2idx = vocab._token2idx
    for bid, (x, query_kw, resp_kw) in enumerate(data):
        inp.append(x[:-1])
        src_attn_truth.append(query_kw)
        truth.append(x[1:])
        msk.append([1 for i in range(len(x) - 1)])
        src_lens.append(len(query_kw))
        n_used_resp_kw = int(len(resp_kw) * 0.8)
        for kw in resp_kw[:n_used_resp_kw]:
            if kw in token2idx:
                wid = token2idx[kw]
                kw_resp_truth[bid, wid] = 1.0
    #print("kw_resp_truth:", kw_resp_truth)
    max_src_len = max(src_lens)
    msk_src = torch.zeros(bsz, max_src_len)
    for i in range(bsz):
        msk_src[i, :src_lens[i]] = torch.ones(src_lens[i]).float()
    # transpose the data
    truth = torch.LongTensor(ListsToTensor(truth, vocab)).t_().contiguous()
    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    src_attn_truth = torch.FloatTensor(reformat(src_attn_truth)).contiguous()
    kw_resp_truth = torch.FloatTensor(kw_resp_truth).contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    # no need to transpose
    msk_src = torch.FloatTensor(msk_src).contiguous()
    return inp, truth, src_attn_truth, kw_resp_truth, msk, msk_src, src_lens


def s2t(strs, vocab):
    inp, msk = [], []
    for x in strs:
        inp.append([w for w in x])
        msk.append([1 for i in range(len(x))])

    inp = torch.LongTensor(ListsToTensor(inp, vocab)).t_().contiguous()
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()
    return inp, msk


class ValidateDataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):

        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = []
        for line in lines[:-1]:  # the last sent may be imcomplete
            qr, kw = line.strip().split('\t')
            tokens = qr.split()
            kw_tokens = kw.split(' ')
            if tokens and kw_tokens:
                data.append((tokens, kw_tokens))
        random.shuffle(data)

        idx = 0
        while idx < len(data):
            yield batchify_original(data[idx:idx + self.batch_size], self.vocab)
            idx += self.batch_size


class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        """

        :param vocab: vocabulary
        :param filename: filename of training data
        :param batch_size: batch size
        :param max_len: maximum sequence length
        """
        print("Use file %s..." % filename)
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

        # added by lixin
        self.query2resp_kw = {}
        self.stopwords = {}
        with open('stopwords.txt', 'r', encoding='UTF-8') as fp:
            for line in fp:
                w = line.strip()
                if w not in self.stopwords:
                    self.stopwords[w] = True
        with open("./data/train_resp_kw.txt", 'r', encoding='UTF-8') as fp:
            for line in fp:
                #print(line)
                eles = line.strip().split("\t")
                if len(eles) == 2:
                    query, kw_string = eles
                if len(eles) == 1:
                    query = eles[0]
                    self.query2resp_kw[query] = []
                kws = kw_string.split(' ')
                key_chars = []
                for w in kws:
                    chars = list(w)
                    use_this_word = True
                    for ch in chars:
                        if ch in self.stopwords:
                            use_this_word = False
                            break
                    if use_this_word:
                        key_chars.extend(chars)
                self.query2resp_kw[query] = list(set(key_chars))
        # added by lixin

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        data = []
        for line in lines[:-1]: #the last sent may be incomplete
            qr, kw = line.strip().split('\t')
            tokens = qr.split()
            bid = tokens.index('<bos>')
            query = ''.join(tokens[:bid])
            resp_kw_tokens = self.query2resp_kw[query]
            random.shuffle(resp_kw_tokens)
            kw_tokens = kw.split()
            if tokens and kw_tokens:
                data.append((tokens, kw_tokens, resp_kw_tokens))
        random.shuffle(data)

        idx = 0
        #print("Number of training data:", len(data))
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None, model_path=''):
        idx2token = [PAD, UNK, BOS, EOS] + (specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
            if token == BOS or token == EOS:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size-1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)
