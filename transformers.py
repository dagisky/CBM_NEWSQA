import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from typing import Dict, Any
from Utils.utils import MLP, OptionalLayer, LayerNorm1
import numpy as np

def save_npy(data, filename):
    np.save("vis/"+filename, data)

class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
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



class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

        # TODO: remove unused entity head?
        self.e = MLP(D_in=config["symbol_size"], hidden_dims=[self.hidden_size, self.hidden_size], D_out=config["vocab_size"])
        

    def forward(self, query_embed: torch.Tensor, TPR: torch.Tensor):     
        e = torch.einsum('bsf->bf', torch.einsum('bsf,bf->bsf', TPR, query_embed))
        e1 = self.e(e)
        
        return e1 # logits
            


def clones(module, N):
    """Produce N identical layers. (in the paper N=6)"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        if int(os.environ["attention_vis"]):
            print("visualizing attention")
            save_npy( self.attn.cpu().detach().numpy(), "attention_e"+str(os.environ["epoch"])+"_t"+str(os.environ["task"])+".npy")
            os.environ["attention_vis"] = "0"
         
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, embedding, position):
        super(EncoderDecoder, self).__init__()
        self.position = position
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        
    def forward(self, src, src_mask, query):
        "Take in and process masked src and target sequences."
        src_embed, query_embed = self.embedding(src, query)
        src_embed = self.position(src_embed)
        
        out = self.encode(src_embed, None)
        return self.decode(out, query_embed)
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, memory, query):
        return self.decoder(query, memory) # the memory is the context


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, relational, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.relational = relational
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.relational)





class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))





class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
