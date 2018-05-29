import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm
import pickle

#Layers adapted for captioning from https://arxiv.org/abs/1705.03122
def Conv1d(in_channels, out_channels, kernel_size, padding, dropout=0):
    m = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # m.weight.data.normal_(0, 0.1)
    worddict_tmp = pickle.load(open('./data/wordlist.p', 'rb'))
    wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
    wordlist = ['EOS'] + sorted(wordlist)
    assert len(wordlist) == num_embeddings
    print '[DEBUG] Load pre-trained wordemb'
    with open('./data/word_emb.txt', 'r') as f:
        emb_tmp = f.read().splitlines()[2:]
    # emb = torch.zeros([len(num_embeddings, embed_dim)])
    for i in tqdm(emb_tmp):
        word = i.split()[0]
        # emb[wordlist.index(word), :] = i.split()[1:]
        m.weight.data[wordlist.index(word), :] = i.split()[1:]
    return m

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class AttentionLayer(nn.Module):
  def __init__(self, conv_channels, embed_dim):
    super(AttentionLayer, self).__init__()
    # print 'Att conv_channels:', conv_channels # 512
    # print 'Att embed_dim:', embed_dim # 512
    self.in_projection = Linear(conv_channels, embed_dim)
    self.out_projection = Linear(embed_dim, conv_channels)
    # self.img_projection = Linear(nattentionfeats, embed_dim)
    self.bmm = torch.bmm

  def forward(self, x, wordemb, imgsfeats):
    residual = x

    # print 'Att x:', x.size() # 15L, 512L
    # print 'Att wordemb:', wordemb.size() # 15L, 512L
    # print 'Att in_projection:', self.in_projection(x).size() # 15L, 512L
    x = (self.in_projection(x) + self.in_projection(wordemb)) * math.sqrt(0.5)
    # print 'Att x:', x.size() # 15L, 512L

    b, c, f_h, f_w = imgsfeats.size()
    y = imgsfeats.view(b, c, f_h*f_w)
    # print 'Att y:', y.size() # 2048L, 49L for resnet , 512L, 49L for vgg16
    # batch1:b*n*m, batch2:b*m*p, output:b*n*p
    x = self.bmm(x, y)

    sz = x.size()
    x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
    x = x.view(sz)
    attn_scores = x

    y = y.permute(0, 2, 1)

    x = self.bmm(x, y)

    s = y.size(1)
    x = x * (s * math.sqrt(1.0 / s))

    x = (self.out_projection(x) + residual) * math.sqrt(0.5)

    return x, attn_scores

class convcap(nn.Module):
  
  def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=.1):
    super(convcap, self).__init__()
    self.nimgfeats = 1000 # 4096
    self.nattentionfeats = 2048
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout 
 
    self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)
    self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

    self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
    self.resproj = Linear(nfeats*2, self.nfeats, dropout=dropout)

    n_in = 2*self.nfeats 
    n_out = self.nfeats
    self.n_layers = num_layers
    self.convs = nn.ModuleList()
    self.attention = nn.ModuleList()
    self.kernel_size = 5
    self.pad = self.kernel_size - 1
    for i in range(self.n_layers):
      self.convs.append(Conv1d(n_in, 2*n_out, self.kernel_size, self.pad, dropout))
      if(self.is_attention):
        self.attention.append(AttentionLayer(n_out, self.nattentionfeats))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
    self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)

  def forward(self, imgsfeats, imgsfc7, wordclass):
    attn_buffer = None
    wordemb = self.emb_0(wordclass)
    wordemb = self.emb_1(wordemb)
    # print 'wordemb:', wordemb.size() # 15L, 512L
    x = wordemb.transpose(2, 1)   
    # print 'x:', x.size() # 512L, 15L
    batchsize, wordembdim, maxtokens = x.size()

    # print 'imgsfc7:', imgsfc7.size() # 1000L
    y = F.relu(self.imgproj(imgsfc7))
    y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
    # print 'y:', y.size() # 512L, 15L
    x = torch.cat([x, y], 1)
    # print 'x<-cat(x,y):', x.size() # 1024L, 15L

    for i, conv in enumerate(self.convs):
      
      if(i == 0):
        x = x.transpose(2, 1)
        residual = self.resproj(x)
        residual = residual.transpose(2, 1)
        x = x.transpose(2, 1)
      else:
        residual = x
      # print 'layer:', i
      # print 'x:', x.size() # 1024L, 15L in layer 0, 512L, 15L in other layer
      # print 'residual:', residual.size() # 512L, 15L

      x = F.dropout(x, p=self.dropout, training=self.training)

      x = conv(x)
      # print 'conv(x):', x.size() # 1024L, 19L
      x = x[:,:,:-self.pad]
      # print 'unpad x:', x.size() # 1024L, 15L

      x = F.glu(x, dim=1)
      # print 'glu(x):', x.size() # 512L, 15L

      if(self.is_attention):
        attn = self.attention[i]
        x = x.transpose(2, 1)
        # print 'imgsfeats:', imgsfeats.size() # 512L, 7L, 7L for vgg16
        x, attn_buffer = attn(x, wordemb, imgsfeats)
        x = x.transpose(2, 1)
        # print 'x, attn:', x.size(), attn_buffer.size() # 512L, 15L ; 15L 49L for vgg
    
      x = (x+residual)*math.sqrt(.5)

    x = x.transpose(2, 1)
  
    x = self.classifier_0(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.classifier_1(x)

    x = x.transpose(2, 1)

    return x, attn_buffer
