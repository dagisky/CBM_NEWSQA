import argparse
import os
import gc
import torch
from torch import nn
# from torch import optim
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from time import gmtime, strftime
from MultiHeadRln import EncoderModel
from Utils.utils import Logger
from data import load_data, convert_examples_to_features
import json
import torch.nn as nn
from Utils.vis import visualize
import collections
from pytorch_transformers import BertTokenizer, BertModel #*
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, random_split
from tqdm import tqdm

gpu_list = [6, 7, 8] # 6, 7 # List of GPU cards to run on [4, 6, 7]
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4" 


def train(model, x, mask, p1, p2, seqlens, mloss, optim, args):
    pred, std = model(torch.transpose(x[0], -2,-1), mask, seqlens) 
    
    optim.zero_grad()
    # print([pred[0].device, p1.device, p1.to(args.device).device])
    loss = mloss(pred[0], p1.to(args.device)) + mloss(pred[1], p2.to(args.device))
    

    std = torch.sum(std)

    loss = loss + args.std_alpha*std
    
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optim.step()

    return loss, pred

def test(model, x, mask, p1, p2, seqlens, mloss, args, visualize=False):
    if visualize:
        attn, std = model(x, mask, seqlens, visualize)
        return attn
    pred, _ = model(torch.transpose(x[0],-2, -1), mask, seqlens)
    loss = mloss(pred[0], p1.to(args.device)) + mloss(pred[1], p2.to(args.device))
    return loss, pred

def get_ans_range(ans):
    if ans[0] == 0 and ans[1] == 0:
        return [0]
    elif ans[0] > ans[1]:
        return [0]
    else:
        return list(range(ans[0], ans[1]+1))

def compute_em(target, pred, batch_size = 0):
    def em(a_gold, a_pred):
        return int(a_gold==a_pred)
    if batch_size == 0:
        return em(target, pred)
    else:
        tot_em = 0
        for t, p in zip(target, pred):
            tot_em += em(t,p)
        return tot_em/batch_size

def compute_f1(target, pred, batch_size=0):
    def f1(a_gold, a_pred):
        common = collections.Counter(a_gold) & collections.Counter(a_pred)
        num_same = sum(common.values())
        if len(a_gold) == 0 or len(a_pred) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(a_gold == a_pred)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same/len(a_pred)
        recall = 1.0 * num_same/len(a_gold)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    if batch_size == 0:
        return f1(target, pred)
    else:
        tot_f1 = 0
        for t, p in zip(target, pred):
            tot_f1 += f1(t,p)
        return tot_f1/batch_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default=True)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--std-alpha', default=0.003, type=float)
    parser.add_argument('--test-split', default=0.1, type=float)
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--exp-decay-rate', default=0.99, type=float)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--hidden-size', default=767, type=int)
    parser.add_argument('--learning-rate', default=0.05, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--model-config', default='rnn_config.ini')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
    parser.add_argument('--model-name', default='MultiHeadRln')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--resume', default='MultiHeadRln_18 _300.model', type=str, metavar='PATH', help='path saved params')
    parser.add_argument("--output_dir", default='./checkpoints/', type=str, 
                        help="The output directory where the model checkpoints will be written.")
    # [/mnt/disk/dagi/BiDAF_multiHead/checkpoints/], [./checkpoints/]
    args = parser.parse_args()
    device = torch.device(f"cuda:{str(gpu_list[0])}" if args.use_cuda and torch.cuda.is_available() else "cpu")
    

    print('loading NewsQA data...')
    setattr(args, 'device', device)
    setattr(args, 'model_time', strftime('%H_%M_%S', gmtime()))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    data = load_data(story_path='../data', question_filename='../data/newsqa-data-v1')
    features = convert_examples_to_features(data, tokenizer, 350, 150, 50, True)
    print("Data loading finished...")
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams']  
   
 
    model = EncoderModel(args, hyp).to(device)
    model = nn.DataParallel(model, device_ids=gpu_list)

    model_loss = nn.NLLLoss()
    optimizer  = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    print("Loading data to RAM: (6: items)")
    all_input_ids = torch.tensor([f.input_ids for f in tqdm(features)], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in tqdm(features)], dtype=torch.long)
    all_seq_lengths = torch.tensor([len(f.input_ids) for f in tqdm(features)], dtype=torch.long)
    all_start_pos = torch.tensor([f.start_position for f in tqdm(features)], dtype=torch.long)
    all_end_pos = torch.tensor([f.end_position for f in tqdm(features)], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in tqdm(features)], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_start_pos, all_end_pos, all_seq_lengths, all_segment_ids, all_example_index)

    lengths = [int(len(dataset)*0.8), int(len(dataset))-int(len(dataset)*0.8)]
    train_dataset, test_dataset = random_split(dataset, lengths)
    print(f'Training Dataset: {int(len(dataset)*0.8)},  Dev Dataset: {int(len(dataset))-int(len(dataset)*0.8)}')
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

    if os.path.exists(f'model/{args.resume}'):
        print(f'loading ..........model/{args.resume}')
        model.load_state_dict(torch.load(f'model/{args.resume}', map_location=args.device)) #  map_location
    # loss, acc, total = 0, 0, 0
    for e in range(args.epoch_start, args.epoch):        
        f1, em, total, loss = 0.0, 0.0, 0.0, 0.0
        for i, batch in enumerate(train_dataloader): 
            batch_loss, out = train(model, bert_model(batch[0]), batch[1], batch[2], batch[3], batch[4], model_loss, optimizer, args)
            pred = list(zip(torch.argmax(out[0], dim=1).cpu().numpy(), torch.argmax(out[1],dim=1).cpu().numpy()))
            ans = list(zip(batch[2].cpu().numpy(), batch[3].cpu().numpy()))
            batch_f1 = compute_f1(ans, pred, len(pred))
            batch_em = compute_em(ans, pred, len(pred))
            total = i
            f1 += batch_f1
            em += batch_em
            loss += float(batch_loss)
            if i%10 == 0: # print every 10th loop
                print('F1:-'+str(f1/(i+1))+'  EM:-'+str(em/(i+1))+'  Loss:'+str(loss/(i+1)))
                                    
        print(f'epoch: {e} /loss: {loss/total:.3f} /F1: {f1/total:.3f} /EM: {em/total:.3f}')
    
    
        dev_f1, dev_em, dev_total, dev_loss = 0.0, 0.0, 0.0, 0.0
        for i,  batch in enumerate(eval_dataloader): #-----
            batch_loass, out = test(model, bert_model(batch[0]), batch[1], batch[2], batch[3], batch[4], model_loss, args)
            pred = list(zip(torch.argmax(out[0], dim=1).cpu().numpy(), torch.argmax(out[1], dim=1).cpu().numpy()))
            ans = list(zip(batch[2].cpu().numpy(), batch[3].cpu().numpy()))
            dev_f1 += compute_f1(ans, pred, len(pred))
            dev_em += compute_em(ans, pred, len(pred))
            dev_loss += float(batch_loss)
            dev_total += i
        print(f'Test:- loss:{dev_loss/dev_total:.3f} /F1: {dev_f1/dev_total} /EM: {dev_em/dev_total}')
           
    #             seqlens = batch.dialog[1].cpu().numpy()
    #             max_seq = batch.dialog[0].size(1) 
    #             mask = list()
    #             for l in seqlens:
    #                 mask.append(torch.cat((torch.ones(l, device=args.device), torch.zeros(max_seq-l, device=args.device)), 0).unsqueeze(0))
    #             mask = torch.cat(mask, 0)

    #             batch_loss, out = test(model, batch.dialog[0], batch.answer, mask, batch.dialog[1], model_loss)            
    #             batch_pred = torch.argmax(out, dim=1)
    #             batch_acc = (batch_pred == batch.answer).sum()
    #             dev_loss += float(batch_loss)
    #             dev_acc += batch_acc.data.item()
    #             dev_total += args.dev_batch_size
    #         print(f'dev-epoch: {e} /dev-loss: {dev_loss/dev_total:.3f} / dev-accuracy: {(dev_acc/dev_total)*100:.3f}%')

    #     if e % 100 == 0:
    #         print('+++++++++++ TEST VIS STARTED ++++++++++++')
    #         # for batch in data.train_iter:
    #         #     seqlen = batch.dialog[1][0].cpu().numpy()
    #         #     max_seq = batch.dialog[0][0].size(0)

                

    #         #     dialog_item, answer_item, mask_item = batch.dialog[0][0], batch.answer[0], batch.dialog[0][1]
    #         #     dialog_item, answer_item = dialog_item.unsqueeze(0),answer_item.unsqueeze(0) 
    #         #     mask = torch.cat((torch.ones(seqlen, device=args.device), torch.zeros(max_seq-seqlen, device=args.device)), 0).unsqueeze(0)                       
                
              
    #         #     attention_values = test(model, dialog_item, answer_item, mask, mask_item.unsqueeze(0), model_loss, True)
    #         #     # attention_values = torch.cat(attention_values, dim=0)
    #         #     dialog_item = dialog_item.cpu().numpy()
    #         #     dialog = list()

    #         #     for idx in dialog_item[0]:
    #         #         dialog.append(data.DIALOG_QUERY.vocab.itos[idx])
    #         #     visualize(attention_values, dialog, args.model_name+'_'+args.task+'_'+str(e))
    #         #     break

    #         logger = Logger('runs/log_' + str(e))
    #         for name, value in model.named_parameters():
    #             # if value.grad is not None:
    #             logger.histo_summary(tag = name+'/-weight', values=value.data.cpu().numpy(), step= e*len(data.train_iter)) 
    #             logger.histo_summary(tag = name+'/-grad', values=value.grad.cpu().numpy(), step= e*len(data.train_iter))
                
    #         torch.save(model.state_dict(), f'model/{args.model_name}_{args.task}_{e}.model')
    #     gc.collect()

    # print('data loading complete!')

    

if __name__ == '__main__':
    main()
