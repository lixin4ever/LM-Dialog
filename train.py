# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from biglm import BIGLM, BIGLM_LSTM
from data import Vocab, DataLoader, ValidateDataLoader
from adam import AdamWeightDecayOptimizer

import argparse, os
import random
import time


def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    #lm_args.dropout = 0.001
    lm_vocab = Vocab(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = BIGLM(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers, lm_args.approx)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.cuda(device)
    return lm_model, lm_vocab, lm_args


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--min_occur_cnt', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument('--valid_every', type=int)
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--approx', type=str, default='none')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)
    parser.add_argument('--backend', type=str)

    parser.add_argument('--lm_type', type=str, default='transformer')
    parser.add_argument('--pretrained_model', type=str)
    parser.add_argument('--use_src_attn', type=int)
    parser.add_argument('--use_resp_kw', type=int)

    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def run(args, local_rank):
    """ Distributed Synchronous """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if (args.world_size == 1 or dist.get_rank() == 0):
        print (vocab.size)
    # added by lixin
    mode2string = {
        0: "basic transformer, use nothing",
        1: "use response keyword",
        2: "use source attention",
        3: "use response keyword + source attention",
        4: "lstm language model with self attention"
    }
    if args.use_src_attn and args.use_resp_kw:
        mode = 3
    elif args.use_src_attn and not args.use_resp_kw:
        mode = 2
    elif not args.use_src_attn and args.use_resp_kw:
        mode = 1
    else:
        mode = 0

    if args.lm_type == 'transformer':
        # transformer language model
        model = BIGLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim, args.num_heads,
                      args.dropout, args.layers, args.approx, mode=mode)
    elif args.lm_type == 'lstm':
        # lstm language model
        model = BIGLM_LSTM(local_rank=local_rank, vocab=vocab, embed_dim=args.embed_dim, num_heads=args.num_heads,
                           dropout=args.dropout, layers=args.layers)
        mode = 4
    else:
        raise Exception("Invalid type %s of language model!!!" % args.lm_type)
    log_fp = open('./log/model%s_log%s' % (mode, local_rank), 'a')

    # added by lixin

    model_string = str(model)
    if local_rank == 0:
        print(model_string)
    model_description = "[Rank %s] Model: %s..." % (local_rank, mode2string[mode])
    print(model_description)

    # add the content to the model file
    log_fp.write(model_string+'\n')
    log_fp.write(model_description+'\n')

    smaller_lr = False
    if args.start_from is not None:
        # continue training from the specified checkpoint
        model_file = './ckpt/%s' % args.start_from
        _, _, epoch_str, batch_str = args.start_from.split('_')
        smaller_lr = True
        batch_acm = int(batch_str.split('batch')[1])
        n_iter = int(epoch_str.split('epoch')[1])
        ckpt = torch.load(model_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        training_string = "[Rank %s] Continue the training of model %s..." % (local_rank, args.start_from)
    elif args.pretrained_model is not None:
        # fine-tuning pretrained model on the given dataset
        model_file = './finetune/%s' % args.pretrained_model
        ckpt = torch.load(model_file, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        batch_acm = 0
        n_iter = 0
        training_string = "[Rank %s] Finetuning model %s on the downstream dataset..." % (local_rank, args.pretrained_model)
    else:
        batch_acm = 0
        n_iter = 0
        training_string = "[Rank %s] Training from scratch..." % local_rank
    print(training_string)
    log_fp.write(training_string+'\n')
    model = model.cuda(local_rank)

    weight_decay_params = []
    no_weight_decay_params = []
    
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay':0.01},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    if args.world_size > 1:
        torch.manual_seed(1234 + dist.get_rank())
        random.seed(5678 + dist.get_rank())
    
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        optimizer = FusedAdam(grouped_params,
                              lr=args.lr,
                              betas=(0.9, 0.999),
                              eps =1e-6,
                              bias_correction=False,
                              max_grad_norm=1.0)
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    else:
        if smaller_lr:
            # halve the learning rate if this flag is true
            args.lr /= 2.0
        optimizer = AdamWeightDecayOptimizer(grouped_params,
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    if args.start_from is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        for pg in optimizer.param_groups:
            pg['lr'] /= 2.0
    init_lr_string = "[Rank %s] Current learning rate: %s" % (local_rank, optimizer.param_groups[0]['lr'])
    log_fp.write(init_lr_string+'\n')
    print(init_lr_string)
    if args.world_size == 1:
        train_data = DataLoader(vocab, args.train_data, args.batch_size, args.max_len)
    else:
        train_data = DataLoader(vocab, './data/train_%s.txt' % local_rank, args.batch_size, args.max_len)
    val_data = ValidateDataLoader(vocab, args.val_data, args.batch_size, args.max_len)
    #batch_acm = 0
    train_data.epoch_id = n_iter
    acc_acm, ntokens_acm, npairs_acm, loss_acm = 0., 0., 0., 0.
    best_val_loss = 99999999.0
    #n_iter = 0
    init_lr = args.lr
    epoch_string = "[Rank %s] Epoch %s (lr=%s):" % (local_rank, n_iter, optimizer.param_groups[0]['lr'])
    log_fp.write(epoch_string)
    print(epoch_string)

    # empty the buffer
    log_fp.flush()

    while True:
        model.train()
        n_train_batches = 0
        #print("Epoch %s:" % n_iter)
        for inp, truth, src_attn_truth, kw_resp_truth, msk, msk_src, src_lens in train_data:
            #batch_acm += 1
            n_train_batches += 1
            #if n_train_batches <= 49000:
            #    continue
            batch_acm += 1
            if batch_acm <= args.warmup_steps:
                update_lr(optimizer, args.lr*batch_acm/args.warmup_steps)
            truth = truth.cuda(local_rank)
            inp = inp.cuda(local_rank)
            msk = msk.cuda(local_rank)

            src_attn_truth = src_attn_truth.cuda(local_rank)
            kw_resp_truth = kw_resp_truth.cuda(local_rank)
            msk_src = msk_src.cuda(local_rank)

            optimizer.zero_grad()
            beg = time.time()
            if args.lm_type == 'transformer':
                # transformer auto-regressive language model
                res, loss, acc, ntokens, npairs = model(inp, truth, src_attn_truth, kw_resp_truth, msk, msk_src, src_lens, args, local_rank)
            else:
                # lstm auto-regressive language model
                res, loss, acc, ntokens, npairs = model(inp=inp, truth=truth, msk=msk)
            end = time.time()
            #print("Time cost: %.2f" % (end - beg))
            loss_acm += loss.item()
            acc_acm += acc
            ntokens_acm += ntokens
            npairs_acm += npairs
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if args.world_size > 1:
                average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #if (args.world_size==1 or dist.get_rank() ==0) and batch_acm%args.print_every == -1%args.print_every:
            if batch_acm%args.print_every == -1%args.print_every:
                print_string = '\t[Rank %s] batch_acm %d, loss %.3f, acc %.3f, x_acm %d'%(local_rank, batch_acm, loss_acm/args.print_every, acc_acm/ntokens_acm, npairs_acm)
                log_fp.write(print_string+'\n')
                print(print_string)
                log_fp.flush()
                acc_acm, ntokens_acm, loss_acm = 0., 0., 0.

            # alternative for validation
            """
            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                if args.pretrained_model is not None:
                    filename = '%s/finetune_model%s_layer%s_epoch%d_batch%d' % (
                    args.save_dir, mode, args.layers, train_data.epoch_id, batch_acm)
                else:
                    filename = '%s/model%s_layer%s_epoch%d_batch%d' % (args.save_dir, mode, args.layers,
                                                                          train_data.epoch_id, batch_acm)

                save_string = "\t\t[Rank %s] save the model to %s..." % (local_rank, filename)
                log_fp.write(save_string+'\n')
                print(save_string)
                log_fp.flush()
                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            """
            # evaluate the trained model on the dev set every N steps
            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.valid_every == -1 % args.valid_every:
                model.eval()
                val_loss_acm = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for val_inp, val_truth, val_src_attn_truth, val_msk, val_msk_src, val_src_lens in val_data:
                        n_val_batches += 1
                        val_truth = val_truth.cuda(local_rank)
                        val_inp = val_inp.cuda(local_rank)
                        val_msk = val_msk.cuda(local_rank)
                        val_src_attn_truth = val_src_attn_truth.cuda(local_rank)
                        val_msk_src = val_msk_src.cuda(local_rank)
                        if args.lm_type == 'transformer':
                            val_res, val_loss, val_acc, val_ntokens, val_npairs = model(
                                inp=val_inp, truth=val_truth, src_attn_truth=val_src_attn_truth, kw_resp_truth=None,
                                msk=val_msk, msk_src=val_msk_src, src_lens=val_src_lens, args=args, local_rank=local_rank)
                        else:
                            val_res, val_loss, val_acc, val_ntokens, val_npairs = model(inp=val_inp, truth=val_truth, msk=val_msk)
                        val_loss_acm += val_loss.item()
                    val_loss_acm /= float(n_val_batches)
                    val_loss_string = "\t[Rank %s] Val loss: %s" % (local_rank, val_loss_acm)
                    print(val_loss_string)
                    log_fp.write(val_loss_string + '\n')
                    log_fp.flush()
                    if val_loss_acm < best_val_loss or True:
                        # save the model
                        best_val_loss = val_loss_acm
                        if args.pretrained_model is not None:
                            filename = '%s/finetune_model%s_layer%s_epoch%d_batch%d' % (
                                args.save_dir, mode, args.layers, train_data.epoch_id, batch_acm)
                        else:
                            filename = '%s/model%s_layer%s_epoch%d_batch%d' % (args.save_dir, mode, args.layers,
                                                                               train_data.epoch_id, batch_acm)
                        exceed_string = "\t[Rank %s] Exceed! Save the model to %s..." % (local_rank, filename)
                        print(exceed_string)
                        log_fp.write(exceed_string + '\n')
                        log_fp.flush()
                        if not os.path.exists(args.save_dir):
                            os.mkdir(args.save_dir)
                        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                                   filename)
                model.train()

        if train_data.epoch_id != n_iter:
            """
            # do the validation
            #inp, truth, src_attn_truth, kw_resp_truth, msk, msk_src, src_lens
            if args.world_size == 1 or dist.get_rank() == 0:
                model.eval()
                val_loss_acm = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for val_truth, val_inp, val_msk in val_data:
                        n_val_batches += 1
                        val_truth = val_truth.cuda(local_rank)
                        val_inp = val_inp.cuda(local_rank)
                        val_msk = val_msk.cuda(local_rank)
                        val_res, val_loss, val_acc, val_ntokens, val_npairs = model(
                            inp=val_inp, truth=val_truth, src_attn_truth=None, kw_resp_truth=None,
                            msk=val_msk, msk_src=None, src_lens=None, args=args, local_rank=local_rank)
                        val_loss_acm += val_loss.item()
                    val_loss_acm /= float(n_val_batches)
                    val_loss_string = "\t[Rank %s] Val loss: %s" % (local_rank, val_loss_acm)
                    print(val_loss_string)
                    log_fp.write(val_loss_string+'\n')
                    log_fp.flush()
                    if val_loss_acm < best_val_loss:
                        # save the model
                        best_val_loss = val_loss_acm
                        if args.pretrained_model is not None:
                            filename = '%s/finetune_model%s_layer%s_epoch%d_batch%d' % (
                                args.save_dir, mode, args.layers, train_data.epoch_id, batch_acm)
                        else:
                            filename = '%s/model%s_layer%s_epoch%d_batch%d' % (args.save_dir, mode, args.layers,
                                                                               train_data.epoch_id, batch_acm)
                        exceed_string = "\t[Rank %s] Exceed! Save the model to %s..." % (local_rank, filename)
                        print(exceed_string)
                        log_fp.write(exceed_string+'\n')
                        log_fp.flush()
                        if not os.path.exists(args.save_dir):
                            os.mkdir(args.save_dir)
                        torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            #if (args.world_size==1 or dist.get_rank() ==0) and batch_acm%args.save_every == -1%args.save_every:
            #    if not os.path.exists(args.save_dir):
            #        os.mkdir(args.save_dir)
            #    torch.save({'args':args, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, '%s/epoch%d_batch_%d'%(args.save_dir, train_data.epoch_id, batch_acm))
            #n_iter += 1
            """
            # learning rate decay
            n_iter = train_data.epoch_id
            if n_iter >= 10 and n_iter % 5 == 0:
                decayed_lr = optimizer.param_groups[0]['lr'] * 0.5
                for pg in optimizer.param_groups:
                    # update lr
                    pg['lr'] = decayed_lr
            cur_lr = optimizer.param_groups[0]['lr']
            epoch_string = "[Rank %s] Epoch %s (lr=%s):" % (local_rank, n_iter, cur_lr)
            log_fp.write(epoch_string+'\n')
            print(epoch_string)
            log_fp.flush()
        # empty the writing buffer


def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    if args.world_size == 1:
        run(args, 0)
        exit(0)

    # added by lixin
    data_lines = []
    with open(args.train_data, 'r', encoding='UTF-8') as fp:
        for line in fp:
            data_lines.append(line)
    random.shuffle(data_lines)
    data_size = len(data_lines)
    data_size_per_split = data_size // args.world_size
    print("Split the data for distributed environment...")
    for i in range(args.world_size):
        data_file_i = './data/train_%s.txt' % i
        if not os.path.exists(data_file_i):
            with open(data_file_i, 'w+', encoding='UTF-8') as fp:
                if i == args.world_size-1:
                    fp.writelines(data_lines[i * data_size_per_split:])
                else:
                    fp.writelines(data_lines[i * data_size_per_split:(i + 1) * data_size_per_split])
    # save the log file for each model
    if not os.path.exists('./log'):
        os.mkdir('./log')

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
