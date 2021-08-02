from typing import Dict, Any
import logging
import argparse
import random
import torch
import json
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from MultiHeadRln import EncoderModel
from time import gmtime, strftime
from dataset.babi.babi import BABI

logger = logging.getLogger(__name__)

gpu_list = [6, 7, 8] # 6, 7 # List of GPU cards to run on [4, 6, 7]
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,4" 



def train(model, story, query, seqlens, answer, mloss, optim, args):


    logits, std = model(story, query, seqlens)    
    optim.zero_grad()
    loss = mloss(logits, answer)
    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
    loss = loss.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
    optimizer.step()

    return loss.item(), correct_batch.item()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default=True)
    parser.add_argument('--path', default='dataset/tasks_1-20_v1-2/en-valid-10k', type=str)
    parser.add_argument('--data-config-file', default='dataset/babi/configs/data_config.json', type=str)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--exp-decay-rate', default=0.99, type=float)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--hidden-size', default=767, type=int)
    parser.add_argument('--learning-rate', default=0.05, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--model-config', default='rnn_config.ini')
    parser.add_argument('--max-gradient-norm', type=float, default=5, help='gradient clipping')
    parser.add_argument('--model-name', default='MultiHeadRln')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--resume', default='MultiHeadRln_18 _300.model', type=str, metavar='PATH', help='path saved params')
    parser.add_argument("--output_dir", default='./checkpoints/', type=str, help="The output directory where the model checkpoints will be written.")

    args = parser.parse_args()
    # device = torch.device(f"cuda:{str(gpu_list[0])}" if args.use_cuda and torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")


    print('loading NewsQA data...')
    setattr(args, 'device', device)
    setattr(args, 'model_time', strftime('%H_%M_%S', gmtime()))

    babi = BABI(args) # data loader size: 
    train_data_loaders = babi.getTrainData()
    print("Data loading finished...")
    with open(args.config) as config_file: 
        hyp = json.load(config_file)['hyperparams']  


    model = EncoderModel(args, hyp).to(device)
    # model = nn.DataParallel(model, device_ids=gpu_list)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    for i in range(args.epochs):
        logging.info(f"##### EPOCH: {i} #####")
        correct = 0
        train_loss = 0
        for k in tqdm(range(babi.num_train_batches)):
            loader_i = random.randint(0,len(babi.train_data_loaders)-1)+1
            try:
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            except StopIteration:
                train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])

            batch_loss, correct_batch = train(model, story, query, story_length, answer, loss_fn, optimizer, args)
            correct += correct_batch
            train_loss += batch_loss
            

            if k % 50 == 0:
                acc = correct / (babi.train_batch_size * (k + 1))
                ls = train_loss / (babi.train_batch_size * (k + 1))
                print(correct)
                print(babi.train_batch_size)
                print(k)
                print(correct / (babi.train_batch_size * (k+1)))
                print(f'Epoch: %d | itt: %d | batch_size: %d => correct: %d  | gross_loss: %d' %(i, k, babi.train_batch_size, correct, train_loss))
                print(f'Epoch: %d | itt: %d  => accuracy: %f  | loss: %f' %(i, k, acc, ls))
        train_acc = correct / (babi.num_train_batches * babi.train_batch_size)
        loss = train_loss / (babi.num_train_batches * babi.train_batch_size)
        print(f'Epoch: %d => accuracy: %d  | loss: %d' %(i, train_acc, loss))

          


