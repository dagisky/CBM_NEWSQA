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

from pytorch_transformers import BertTokenizer, BertModel #*
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

gpu_list = [6, 7, 8] # 6, 7 # List of GPU cards to run on [4, 6, 7]
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4" 


def train(model, x, mask, p1, p2, seqlens, mloss, optim, args):
    pred, std = model(torch.transpose(x[0], -2,-1), mask, seqlens) 
    
    optim.zero_grad()  
    loss = mloss(pred[0], torch.argmax(p1.to(args.device), dim=1)) + mloss(pred[1], torch.argmax(p1.to(args.device), dim=1))
    print(std.size())

    std = torch.sum(std)

    loss = loss + args.std_alpha*std
    
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optim.step()

    return loss, pred

def test(model, x, y, mask, seqlens, mloss, visualize=False):
    if visualize:
        attn, std = model(x, mask, seqlens, visualize)
        return attn
    pred, _ = model(x, mask, seqlens)
    log_softmax = nn.LogSoftmax()
    loss = mloss(log_softmax(pred), y)
    return loss, F.softmax(pred, dim=1)


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

    data = load_data(story_path='../data', question_filename='../data/newsqa-data-v1', size=500)
    features = convert_examples_to_features(data, tokenizer, 200, 100, 50, True)

    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams']  
   
 
    model = EncoderModel(args, hyp).to(device)
    model = nn.DataParallel(model, device_ids=gpu_list)

    model_loss = nn.NLLLoss()
    optimizer  = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([len(f.input_ids) for f in features], dtype=torch.long)
    all_start_pos = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_pos = torch.tensor([f.end_position for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_seq_lengths, all_segment_ids, all_example_index)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    

    if os.path.exists(f'model/{args.resume}'):
        print(f'loading ..........model/{args.resume}')
        model.load_state_dict(torch.load(f'model/{args.resume}', map_location=args.device)) #  map_location
    # loss, acc, total = 0, 0, 0
    for e in range(args.epoch_start, args.epoch):        
        loss, acc, total = 0.0, 0.0, 0.0
        for i, batch in enumerate(eval_dataloader): 
            batch_loss, out = train(model, bert_model(batch[0]), batch[1], batch[3], batch[4], batch[2], model_loss, optimizer, args)
            print(batch_loss)        
            batch_pred_p1 = torch.argmax(out[0], dim=1)
            batch_pred_p2 = torch.argmax(out[1], dim=1)
            loss += float(batch_loss)
            total += args.train_batch_size

        print(f'epoch: {e} /loss: {loss/total:.3f}')
    #     gc.collect()
    #     if e % 10 == 0: # test module 
    #         dev_loss, dev_acc, dev_total = 0, 0, 0
    #         for batch in data.dev_iter: #-----
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
