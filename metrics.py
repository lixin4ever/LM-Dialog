# calculate the BLEU score and the distinct score
import os
import sys


if __name__ == '__main__':
    args = sys.argv
    model = args[1]
    decode_type = args[2]
    k = args[3]
    use_online_test_data = int(args[4])
    # in our test set, there are four references for each query and we place them in four different files
    gold_resp_file1 = './data/gold_resp1.txt'
    gold_resp_file2 = './data/gold_resp2.txt'
    gold_resp_file3 = './data/gold_resp3.txt'
    gold_resp_file4 = './data/gold_resp4.txt'
    if use_online_test_data:
        pred_resp_file = './dec_res/pred_resp_online_%s_%s%s.txt' % (model, decode_type, k)
        pred_dist_file = './dec_res/pred_resp_online_%s_%s%s_dist.txt' % (model, decode_type, k)
    else:
        pred_resp_file = './dec_res/pred_resp_%s_%s%s.txt' % (model, decode_type, k)
        pred_dist_file = './dec_res/pred_resp_%s_%s%s_dist.txt' % (model, decode_type, k)
        os.system('perl multi-bleu.perl %s %s %s %s < %s' % (gold_resp_file1, gold_resp_file2, gold_resp_file3, gold_resp_file4, pred_resp_file))
    # calculate dist metrics
    os.system('python distinct_topk.py 800 < %s > %s' % (pred_resp_file, pred_dist_file))
