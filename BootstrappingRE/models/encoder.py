import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim


class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300, pos_embedding_dim=5,cuda=True):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.cuda=cuda

        # Word embedding
        unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))

        # Position Embedding
        self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        posh = inputs['posh']
        post = inputs['post']

        x = torch.cat([self.word_embedding(word),
                       self.pos1_embedding(posh),
                       self.pos2_embedding(post)], 2)
        return x

class CNN_Encoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300, pos_embedding_dim=5, hidden_size=230,cuda=True):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.cuda=cuda
        self.embedding = Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim,cuda)
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)
        self.model_type="CNN"

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.conv(x.transpose(1, 2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

class PCNN_Encoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300, pos_embedding_dim=5, hidden_size=230,cuda=True):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.cuda = cuda
        self.model_type = "PCNN"
        self.embedding = Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim,cuda)
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

    def forward(self, inputs):
        x,mask = self.embedding(inputs),inputs['mask']
        x = self.conv(x.transpose(1, 2))  # n x hidden x length
        mask = 1 - self.mask_embedding(mask).transpose(1, 2)  # n x 3 x length
        pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1)
        return x.squeeze(2) # n x (hidden_size * 3)

# class BERT_Encoder(nn.Module):