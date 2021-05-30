from multiprocessing import Pool
from collections import Counter
import sys
import re
import argparse
from data import BOS, EOS

BUFSIZE = 100000
MAX_LEN = 512


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--tgt_file', type=str)
    parser.add_argument('--nprocessors', type=int)
    return parser.parse_args()


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def process(doc):
    fs = doc.split("\t")
    if len(fs) == 2:
        q, r = fs[0].strip(), fs[1].strip()
        q = ''.join(q.split())
        r = ''.join(r.split())
        return [w for w in q] + [BOS] + [w for w in r] + [EOS] 
    else:
        return []


def save(cnt, docs, nprocessors, fo):
    res = pool.map(process, docs, len(docs)//nprocessors)
    all_lines = []
    for xs in res:
         all_lines.append(xs)
    
    for x in all_lines:
        cnt.update(x)
        fo.write(' '.join(x)+'\n')


def save2(cnt, docs, fo):
    for doc in docs:
        res = process(doc)
    
        for x in res:
            cnt.update(x)
            fo.write(' '.join(x)+'\n')


if __name__ == "__main__":
    print("start..")
    args = parse_config()

    pool = Pool(args.nprocessors)
    cnt = Counter()
    docs = []
    with open(args.tgt_file, 'w', encoding ='utf8') as fo:
        with open(args.src_file, "r") as fi:
            for line in fi:
                line = line.strip()
                if line:
                    docs.append(line)
                    
                if len(docs) == BUFSIZE:
                    save(cnt, docs, args.nprocessors, fo)                    
                    docs = []
                    print(BUFSIZE)
        if docs:
            save(cnt, docs, args.nprocessors, fo)
            print(len(docs))
    if 'train' in args.tgt_file:
        # only save the vocab built from training dataset
        print("vocab")
        with open('./data/vocab.txt', 'w', encoding ='utf8') as f:
            for x, y in cnt.most_common():
                if x == BOS or x == EOS:
                    continue
                f.write(x + '\t' + str(y) + '\n')
    print("done")
