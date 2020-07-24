import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import configparser
from relational import RelationalLayer


class Attention(nn.Module):

    def __init__(self, input_dim, output_dim):
        """
        Attention Layer for Multi-Head Sequential Attention. The multi-head attention
        runns over squence of text and returns list of weighted attention values.
        Args:
            input_dim (int) : netowrk input dimention
            output_dim (int): network output dimention (equal to seq_len)

        Parameters:
        ------------------------------
            attn (nn.Linear): Lienar module
            v (Tensor): Scaling weight
        """
        super().__init__()        
        self.attn = nn.Linear((input_dim * 6), output_dim)
        self.v = nn.Parameter(torch.rand(output_dim)) 


    def forward(self, hidden, encoder_outputs, mask, c_t):    

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1] 
        c_t = c_t.unsqueeze(1).repeat(1, src_len, 1)
        energy = F.relu(self.attn(torch.cat((hidden, c_t, encoder_outputs), dim=2)))  #relu   
        v = self.v.repeat(batch_size, 1).unsqueeze(2)
        attention = torch.bmm(energy, v).squeeze(2) 
        attention = attention.mul(mask)    
        return F.softmax(attention, dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, device, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        """
        The MultiHeadAttention creates of K number of attention heads that represent concepts
        Args:
            input_din (int): netowrk input dimention
            output_dim (int): network output dimention (equal to seq_len)
            num_heads (int): number of heads int the multi-head attention 
            device (torch.device)
            dropout (float)

        """
        self.num_heads = num_heads
        self.device = device
        self.input_dim = input_dim

        self.attn_heads = nn.ModuleList([
                Attention(input_dim, output_dim) for _ in range(self.num_heads)
        ])
        
        self.rnn = nn.GRU(6*input_dim, input_dim*2, dropout=dropout)

    def forward(self, hidden, encoder_outputs, mask, display_attn=False):
        self.rnn.flatten_parameters()
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)       
        c_t = torch.zeros(batch_size, self.input_dim*2).to(self.device)
        a = list()
        for i, attn in enumerate(self.attn_heads):
            a.append(attn(hidden, encoder_outputs, mask, c_t))
            h = torch.bmm(a[-1].unsqueeze(1), torch.cat((hidden, encoder_outputs), dim=2))
            on = torch.cat((h,c_t.unsqueeze(1)), dim=2)            
            c_t, _ = self.rnn(on.permute(1,0,2))
            c_t = c_t.squeeze(0)            
        attn_track = torch.sum(torch.reshape(torch.cat(a, dim=1), (batch_size, self.num_heads, -1)), dim=1)
        std = torch.sum(torch.std(attn_track, dim=1))     

        if display_attn:
            return a, std
        a = torch.reshape(torch.cat(a, dim=1), (batch_size, self.num_heads, -1))
        return (a), std


class EncoderModel(nn.Module):
    def __init__(self, args, hyp):
        super().__init__()
        """
        The Encoder Contains Word Embedding, LSTM contextual encoder and Multi-Head Attention
        Args:
            Model Parameters: [device (torch.device), dialog_vocab(int), answer_vocab(int), model_configuration]
        
        """

        self.args = args
        self.hyp = hyp
        self.use_cuda = args.use_cuda

        if args.use_cuda:
            print('[ Using CUDA ]')
            torch.cuda.device(args.device)

        self.hidden_size = int(hyp["rnn"]["hidden_size"]) # hidden_size
        self.num_layers = int(hyp["rnn"]["num_layers"]) 
        self.dropout = float(hyp["rnn"]["dropout"]) 
        self.bidirectional = bool(hyp["rnn"]["bidirectional"])

        self.emb_dim = self.hidden_size

        
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_size, self.num_layers, dropout=self.dropout, bidirectional = self.bidirectional)
        
        self.num_attn_heads =  8
        self.multi_head_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, self.num_attn_heads, self.args.device)
        # self.fc = nn.Linear(self.hidden_size * 2 * self.num_attn_heads, args.answer_vocab)
        self.relational = RelationalLayer(self.hidden_size * 2, args.answer_vocab, self.args.device, self.hyp["relational"])
        self.dropout = nn.Dropout(self.dropout)
                
        self.ones = torch.tensor(torch.ones(1), dtype=torch.float, device=args.device, requires_grad=False) #  
        self.zeros = torch.tensor(torch.zeros(1), dtype=torch.float, device=args.device, requires_grad =False)# device=args.device, 


        self.softmax = nn.LogSoftmax()
        # self.loss = nn.NLLLoss()
        self.device = args.device

        self.params = { # NOTE WE CAN TRAIN BY PART i.e first train the other nets next include rln nets
                'embedding':self.embedding,
                'rnn': self.rnn,                
                'multi_head': self.multi_head_attn, #'attn':self.attn,
                'relational': self.relational
                }    
        if self.use_cuda:
            self.cuda()


    def cuda(self):
        for var in self.params.values():
            # var.cuda(device = self.args.device)
            var = var.to(self.args.device)

        
    def forward(self, src, mask, lengths, display_attn = False):          

        self.rnn.flatten_parameters()
        embded = self.dropout(src)        
        # h0, c0 = self.init_zeros(len(src))
        embded = embded.permute(1,0,2)
        pack = nn.utils.rnn.pack_padded_sequence(embded, lengths, batch_first=False, enforce_sorted=True)
        outputs, (hidden, state) = self.rnn(pack) #(h0, c0)
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False, padding_value=0.0)
      

        hidden = hidden[-2:,:,:].permute(1,0,2)
        hidden = torch.reshape(hidden, (hidden.size(0), -1))
        outputs = outputs[0].permute(1,0,2)
        if not display_attn: 
            mask = mask.permute(1, 0)[:outputs.size(1)]
            mask = mask.permute(1, 0)
         
        attn, std = self.multi_head_attn(hidden, outputs, mask, display_attn)

        if display_attn:
            return attn, std
     

        attn = torch.split(attn, 1, dim=1)
        multi_head_weighted = []

        for a in attn: 
            multi_head_weighted.append(torch.bmm(a, outputs))
         
        multi_head_weighted = torch.cat(multi_head_weighted, dim=1)
        out = self.relational(multi_head_weighted)
        return out, std



            
