from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import configparser
from relational import RelationalLayer


class Attention(nn.Module):

    def __init__(self, input_dim, output_dim, growth_factor=2):
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
        self.attn = nn.Linear((input_dim * 3 * growth_factor), output_dim)
        self.v = nn.Parameter(torch.rand(output_dim)) 


    def forward(self, hidden, encoder_outputs, c_t):    

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1] 
        c_t = c_t.unsqueeze(1).repeat(1, src_len, 1)

        energy = F.relu(self.attn(torch.cat((hidden, c_t, encoder_outputs), dim=2)))  #relu   
        v = self.v.repeat(batch_size, 1).unsqueeze(2)
        attention = torch.bmm(energy, v).squeeze(2)    
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

    def forward(self, hidden, encoder_outputs, query, display_attn=False):
        self.rnn.flatten_parameters()
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)      
        c_t = torch.zeros(batch_size, self.input_dim*2)# .cuda(async=True)
        a = list()
        for i, attn in enumerate(self.attn_heads):
            a.append(attn(hidden, encoder_outputs, c_t))
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

        self.hidden_size = int(hyp["input"]["embedding_size"]) # hidden_size
        self.num_layers = int(hyp["rnn"]["num_layers"]) 
        self.dropout = float(hyp["rnn"]["dropout"]) 
        self.bidirectional = bool(hyp["rnn"]["bidirectional"])

        self.emb_dim = self.hidden_size

        self.input_model = InputModule(hyp["input"])
        self.rnn = nn.LSTM(self.emb_dim, self.hidden_size, self.num_layers, dropout=self.dropout, bidirectional = self.bidirectional)
        
        if(self.bidirectional):
            self.lin = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
            self.out_lin = nn.Linear(2*self.hidden_size, hyp["input"]["vocab_size"])
        else:
            self.lin = nn.Linear(self.hidden_size, self.hidden_size)
            self.out_lin = nn.Linear(self.hidden_size, hyp["input"]["vocab_size"])
        
        self.num_attn_heads =  8
        self.multi_head_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, self.num_attn_heads, self.args.device)
        
        
        self.relational = RelationalLayer(self.hidden_size * 2, self.hidden_size * 2 , self.args.device, self.hyp["relational"])
        self.dropout = nn.Dropout(self.dropout)
                
        self.ones = torch.tensor(torch.ones(1), dtype=torch.float, device=args.device, requires_grad=False) #  
        self.zeros = torch.tensor(torch.zeros(1), dtype=torch.float, device=args.device, requires_grad =False)# device=args.device, 


        self.softmax = nn.LogSoftmax()
        self.device = args.device

        self.params = { # NOTE WE CAN TRAIN BY PART i.e first train the other nets next include rln nets
                'rnn': self.rnn,                
                'multi_head': self.multi_head_attn, #'attn':self.attn,
                'relational': self.relational
                }    
        # if self.use_cuda:
        #    self.cuda()


    def cuda(self):
        for var in self.params.values():
            # var.cuda(device = self.args.device)
            var = var.to(self.args.device)

        
    def forward(self, src, query, lengths, display_attn = False): 
        
        src, query = self.input_model(src, query)
        src = src.permute(1,0,2)                 
        
        self.rnn.flatten_parameters()   
        pack = nn.utils.rnn.pack_padded_sequence(src, lengths, batch_first=False, enforce_sorted=False)
        outputs, (hidden, state) = self.rnn(pack) 
        outputs = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False, padding_value=0.0)
        outputs = outputs[0].permute(1,0,2)
        outputs = self.lin(outputs)

        hidden = hidden[-2:,:,:].permute(1,0,2)
        hidden = torch.reshape(hidden, (hidden.size(0), -1))
        
         
        attn, std = self.multi_head_attn(hidden, outputs, query, display_attn)

        if display_attn:
            return attn, std
     

        attn = torch.split(attn, 1, dim=1)
        multi_head_weighted = []

        for a in attn: 
            multi_head_weighted.append(torch.bmm(a, outputs))
         
        multi_head_weighted = torch.cat(multi_head_weighted, dim=1)
        out = self.relational(multi_head_weighted)
        out = self.out_lin(out)
        return self.softmax(out), std


class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"], embedding_dim=config["embedding_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["embedding_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed[:query_embed.shape[1]])
        return sentence_sum, query_sum



# class InferenceModule(nn.Module):
#     def __init__(self, config: Dict[str, Any]):
#         super(InferenceModule, self).__init__()
#         self.hidden_size = config["hidden_size"]
#         self.ent_size = config["entity_size"]
#         self.role_size = config["role_size"]
#         self.symbol_size = config["symbol_size"]
#         # output embeddings
#         self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
#         nn.init.xavier_uniform_(self.Z.data)

#         # TODO: remove unused entity head?
#         self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
#                                     hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
#         self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
#                                     hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
#         self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
#                                      for _ in range(3)]

#     def forward(self, query_embed: torch.Tensor, TPR: torch.Tensor):
#         e1, e2 = [module(query_embed) for module in self.e]
#         r1, r2, r3 = [module(query_embed) for module in self.r]

#         i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
#         i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
#         i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

#         step_sum = i1 + i2 + i3
#         logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
#         return logits



            
