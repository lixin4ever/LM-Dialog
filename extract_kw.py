import jieba
import jieba.analyse
# set stopwords list for jieba keyword extractor
jieba.analyse.set_stop_words(stop_words_path='stopwords.txt')
stopwords = {}


def calculate_pmi(w1, w2):
    try:
        wid1 = vocab[w1]
        wid2 = vocab[w2]
    except KeyError:
        # ignore this response word
        return -999.0
    # doc set for w1
    #doc_idxs_w1 = set(id2doc[wid1])
    doc_idxs_w1 = id2doc[wid1]
    # doc set for w2
    #doc_idxs_w2 = set(id2doc[wid2])
    doc_idxs_w2 = id2doc[wid2]
    # oc-occurrence set for w1 & w2
    joint_set = doc_idxs_w1 & doc_idxs_w2
    pmi = float(len(joint_set)) / (len(doc_idxs_w1) * len(doc_idxs_w2))
    return pmi


with open('stopwords.txt', mode='r', encoding='UTF-8') as fp:
    for line in fp:
        word = line.strip()
        stopwords[word] = True

base_dir = '../dataset/dialog-700w-filtered'

train_file = '%s/train.txt' % base_dir
train_kw_file = '%s/train_kw.txt' % base_dir

queries = []
resps = []

print("Load training data from the disk")
with open(train_file, 'r', encoding='UTF-8') as fp:
    for line in fp:
        query, resp = line.strip().split('\t')
        queries.append(query)
        resps.append(resp)
print("obtain %s queries..." % len(queries))
kw_results = []
count = 0

vocab = {}
# word id and sentence id
wid, sid = 0, 0
id2doc = {}
print("Build the index for calculating p(x, y)....")
query_avg_len = 0.0
resp_avg_len = 0.0
for sid, (query, resp) in enumerate(zip(queries, resps)):
    qws_full = query.split()
    rws_full = resp.split()
    qws = [w for w in qws_full if w not in stopwords]
    rws = [w for w in rws_full if w not in stopwords]
    ws = qws + rws
    for w in ws:
        if w not in vocab:
            vocab[w] = wid
            wid += 1
        cur_wid = vocab[w]
        if cur_wid not in id2doc:
            id2doc[cur_wid] = [sid]
        else:
            id2doc[cur_wid].append(sid)
    query_avg_len += len(qws_full)
    resp_avg_len += len(rws_full)
query_avg_len /= float(len(queries))
resp_avg_len /= float(len(resps))

print("Transform the doc list to doc set...")
for wid in id2doc:
    doc_list = id2doc[wid]
    doc_set = set(doc_list)
    # update from list to set
    id2doc[wid] = doc_set

# number of the keyword candidates
K = 5
empty_count = 0
print("Finding the keywords according to the PMI-based approach...")
for query, resp in zip(queries, resps):
    qws = [w for w in query.split() if w not in stopwords]
    pmi_scores = []
    # keyword from the corresponding response
    resp_kw = jieba.analyse.extract_tags(resp, topK=5)
    for qw in qws:
        # regard the maximum score as the pmi score of the current word with respect to the query
        all_scores = [calculate_pmi(qw, rw) for rw in resp_kw]
        if all_scores == []:
            break
        pmi_scores.append(max(all_scores))
    if pmi_scores == []:
        kw_results.append(jieba.analyse.extract_tags(query, topK=K))
        empty_count += 1
        continue
    results = list(zip(qws, pmi_scores))
    # sort the results in descending order
    results.sort(key=lambda t: t[1], reverse=True)
    # select K candidates with highest pmi scores
    kw_results.append([t[0] for t in results[:K]])
    count += 1
    if count % 10000 == 0:
        print("Processed %s queries..." % count)
print("%s queries not having informative responses..." % empty_count)
with open(train_kw_file, 'w+', encoding='UTF-8') as fp:
    for res in kw_results:
        record = []
        record.extend(res)
        fp.write('\t'.join(record) + '\n')

