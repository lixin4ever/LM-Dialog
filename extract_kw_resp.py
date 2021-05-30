import jieba
import jieba.analyse


train_path = './data/train_with_kw.txt'
stw_path = 'stopwords.txt'
jieba.analyse.set_stop_words(stw_path)

stopwords = {}
with open(stw_path, 'r', encoding='UTF-8') as fp:
    for line in fp:
        w = line.strip()
        if w not in stopwords:
            stopwords[w] = True

query2resps = {}

with open(train_path, 'r', encoding='UTF-8') as fp:
    for line in fp:
        data, query_kw = line.strip().split('\t')
        char_seq = data.split(' ')
        bid = char_seq.index("<bos>")
        eid = char_seq.index("<eos>")
        q = ''.join(char_seq[:bid])
        r = ''.join(char_seq[bid+1:eid])
        if q not in query2resps:
            query2resps[q] = [r]
        else:
            query2resps[q].append(r)

query2kws = {}
count = 0
for q in query2resps:
    responses = query2resps[q]
    kw_all = []
    for r in responses:
        kw_resp = jieba.analyse.extract_tags(r, topK=5)
        kw_all.extend(kw_resp)
        count += 1
        if count % 100000 == 0:
            print("Processed %s queries..." % count)
    query2kws[q] = list(set(kw_all))
    #if count >= 100000:
    #    break

train_resp_kw_path = './data/train_resp_kw.txt'
with open(train_resp_kw_path, 'w+', encoding='UTF-8') as fp:
    lines = []
    for q in query2kws:
        keywords = query2kws[q]
        lines.append('%s\t%s\n' % (q, ' '.join(keywords)))
    fp.writelines(lines)

