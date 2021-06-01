import sys
import numpy as np
sys.path.append('..')
import torch
import random
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import sklearn.metrics
import copy

class Siamese(nn.Module):

    def __init__(self, encoder, hidden_size=230, drop_rate=0.5, pre_rep=None, euc=True):
        nn.Module.__init__(self)
        self.encoder = encoder # Should be different from main sentence encoder
        if encoder.model_type=="CNN": self.hidden_size = hidden_size
        else: self.hidden_size = 3*hidden_size
        # self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.fc2 = nn.Linear(hidden_size * 2, 1)
        if encoder.model_type=="CNN": self.fc = nn.Linear(hidden_size, 1)
        else: self.fc = nn.Linear(hidden_size*3, 1)

        self.criterion = nn.BCELoss(reduction="none")
        self.drop = nn.Dropout(drop_rate)
        self._accuracy = 0.0
        self.pre_rep = pre_rep
        self.euc = euc
        self.model_type="Siamese"

    def forward(self, data, num_size, num_class, threshold=0.5):
        x = self.encoder(data).contiguous().view(num_class, num_size, -1)
        x1 = x[:, :num_size//2].contiguous().view(-1, self.hidden_size)
        x2 = x[:, num_size//2:].contiguous().view(-1, self.hidden_size)
        y1 = x[:num_class//2,:].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class//2:,:].contiguous().view(-1, self.hidden_size)
        # y1 = x[0].contiguous().unsqueeze(0).expand(x.size(0) - 1, -1, -1).contiguous().view(-1, self.hidden_size)
        # y2 = x[1:].contiguous().view(-1, self.hidden_size)

        label = torch.zeros((x1.size(0) + y1.size(0))).long().cuda()
        label[:x1.size(0)] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)

        if self.euc:
            dis = torch.pow(z1 - z2, 2)
            dis = self.drop(dis)
            score = torch.sigmoid(self.fc(dis).squeeze())
        else:
            z = z1 * z2
            z = self.drop(z)
            z = self.fc(z).squeeze()
            # z = torch.cat([z1, z2], -1)
            # z = F.relu(self.fc1(z))
            # z = self.fc2(z).squeeze()
            score = torch.sigmoid(z)

        self._loss = self.criterion(score, label.float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def encode(self, dataset, batch_size=0):
        self.encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'posh' in dataset:
                            _['posh'] = dataset['posh'][scope]
                            _['post'] = dataset['post'][scope]
                        _x = self.encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def forward_infer(self, x, y, threshold=0.5, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < 1] = 0
        pred[pred > 0] = 1
        return pred

    def forward_infer_sort(self, x, y, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = []
        for i in range(score.size(0)):
            pred.append((score[i], i))
        pred.sort(key=lambda x: x[0], reverse=True)
        return pred

class Triplet(nn.Module):

    def __init__(self, encoder, hidden_size=230, drop_rate=0.5, pre_rep=None, euc=True):
        nn.Module.__init__(self)
        self.encoder = encoder # Should be different from main sentence encoder
        if encoder.model_type == "CNN":self.hidden_size = hidden_size
        else:self.hidden_size = 3 * hidden_size
        # self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.fc2 = nn.Linear(hidden_size * 2, 1)
        if encoder.model_type == "CNN": self.fc = nn.Linear(hidden_size, 1)
        else: self.fc = nn.Linear(hidden_size * 3, 1)
        #self.fc1 = nn.Linear(hidden_size, 1)
        #self.fc2 = nn.Linear(hidden_size, 1)
        self.criterion = nn.TripletMarginLoss(reduction="none")
        #self.criterion = nn.MSELoss(reduction="none")
        self.drop = nn.Dropout(drop_rate)
        self._accuracy = 0.0
        self.pre_rep = pre_rep
        self.euc = euc
        self.model_type="Triplet"

    def forward(self, data, num_size, num_class, threshold=0.5):
        x = self.encoder(data).contiguous().view(num_class, num_size, -1)
        x1 = x[:num_class//2, :num_size//2].contiguous().view(-1, self.hidden_size)
        x2 = x[:num_class//2, num_size//2:].contiguous().view(-1, self.hidden_size)
        x3 = x[num_class//2:, :num_size//2].contiguous().view(-1, self.hidden_size)
        y1 = x[num_class//2:,:num_size//2].contiguous().view(-1, self.hidden_size)
        y2 = x[num_class//2:,num_size//2:].contiguous().view(-1, self.hidden_size)
        y3 = x[:num_class//2, num_size//2:].contiguous().view(-1, self.hidden_size)
        # y1 = x[0].contiguous().unsqueeze(0).expand(x.size(0) - 1, -1, -1).contiguous().view(-1, self.hidden_size)
        # y2 = x[1:].contiguous().view(-1, self.hidden_size)
        index = list(range((num_class // 2) * (num_size // 2)))
        random.shuffle(index)
        x3,y3=x3[index],y3[index]

        label = torch.zeros(2*(x1.size(0) + y1.size(0))).long().cuda()
        label[:(x1.size(0) + y1.size(0))] = 1
        z1 = torch.cat([x1, y1], 0)
        z2 = torch.cat([x2, y2], 0)
        z3 = torch.cat([x3, y3], 0)

        d1=torch.cat([z1, z3], 0)
        d2=torch.cat([z2, z2], 0)
        d3 = torch.cat([z3, z1], 0)

        if self.euc:
            #dis=torch.pow(d1 - d2, 2)
            #dis = self.drop(dis)
            #dis_ap,dis_an=torch.pow(z2 - z1, 2),torch.pow(z2 - z3, 2)
            dis_ap_l2,dis_an_l2=F.pairwise_distance(d2,d1, p=2),F.pairwise_distance(d2,d3, p=2)
            dis_ap_l2_exp,dis_an_l2_exp=torch.exp(dis_ap_l2),torch.exp(dis_an_l2)
            dis_ap_softmax,dis_an_softmax=\
                dis_ap_l2_exp/(dis_ap_l2_exp+dis_an_l2_exp),dis_an_l2_exp/(dis_ap_l2_exp+dis_an_l2_exp)
            #dis_pn=torch.pow(z3 - z1, 2)
            #dis_ap,dis_an = self.drop(dis_ap),self.drop(dis_an)
            #score = torch.sigmoid(self.fc(dis).squeeze())
            #score = torch.sigmoid(self.fc(dis_ap).squeeze())
            score=dis_an_softmax
        else:
            dis_ap, dis_an = F.pairwise_distance(z1, z2, p=2), F.pairwise_distance(z3, z2, p=2)
            #dis_ap_l2, dis_an_l2 = F.pairwise_distance(z2, z1, p=2), F.pairwise_distance(z2, z3, p=2)
            #dis_ap_softmax, dis_an_softmax = F.softmax(dis_ap_l2, dim=0), F.softmax(dis_an_l2, dim=0)
            # dis_ap,dis_an = self.drop(dis_ap),self.drop(dis_an)
            dis_ap, dis_an = F.sigmoid(dis_ap), F.sigmoid(dis_an)
            score = torch.pow(dis_ap-dis_an,2)


        #self._loss = self.criterion(dis_ap_softmax, dis_an_softmax).mean()
        self._loss = self.criterion(z2,z1,z3).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        self._accuracy = torch.mean((pred == label).type(torch.FloatTensor))
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        self._prec = float(np.logical_and(pred == 1, label == 1).sum()) / float((pred == 1).sum() + 1)
        self._recall = float(np.logical_and(pred == 1, label == 1).sum()) / float((label == 1).sum() + 1)

    def encode(self, dataset, batch_size=0):
        self.encoder.eval()
        with torch.no_grad():
            if self.pre_rep is not None:
                return self.pre_rep[dataset['id'].view(-1)]

            if batch_size == 0:
                x = self.encoder(dataset)
            else:
                total_length = dataset['word'].size(0)
                max_iter = total_length // batch_size
                if total_length % batch_size != 0:
                    max_iter += 1
                x = []
                for it in range(max_iter):
                    scope = list(range(batch_size * it, min(batch_size * (it + 1), total_length)))
                    with torch.no_grad():
                        _ = {'word': dataset['word'][scope], 'mask': dataset['mask'][scope]}
                        if 'posh' in dataset:
                            _['posh'] = dataset['posh'][scope]
                            _['post'] = dataset['post'][scope]
                        _x = self.encoder(_)
                    x.append(_x.detach())
                x = torch.cat(x, 0)
            return x

    def forward_infer(self, x, y, threshold=0.5, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

        if self.euc:
            dis = torch.pow(x - y, 2)
            #score = torch.sigmoid(self.fc(dis).squeeze(-1)).mean(0)
            score = torch.sigmoid(self.fc1(dis).squeeze(-1)).mean(0)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > threshold] = 1
        pred = pred.view(support_size, -1).sum(0)
        pred[pred < 1] = 0
        pred[pred > 0] = 1
        return pred

    def forward_infer_sort(self, x, y,z, batch_size=0):
        x = self.encode(x, batch_size=batch_size)
        support_size = x.size(0)
        y = self.encode(y, batch_size=batch_size)
        z = self.encode(z, batch_size=batch_size)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        z = z.unsqueeze(1)

        if self.euc:
            dis_ap, dis_an = torch.pow(x - y, 2), torch.pow(z - y, 2)
            # print(dis_ap.size(),dis_an.size())
            dis_ap_l2, dis_an_l2 = torch.sqrt(torch.sum(dis_ap, 2)), torch.sqrt(torch.sum(dis_an, 2))
            # print(dis_ap_l2.size(),dis_an_l2.size())
            dis_ap_l2_mean, dis_an_l2_mean = dis_ap_l2.mean(0), dis_an_l2.mean(0)
            # print(dis_ap_l2_mean.size(),dis_an_l2_mean.size())
            dis_ap_l2_exp, dis_an_l2_exp = torch.exp(dis_ap_l2_mean), torch.exp(dis_an_l2_mean)
            dis_ap_softmax, dis_an_softmax = \
                dis_ap_l2_exp / (dis_ap_l2_exp + dis_an_l2_exp), dis_an_l2_exp / (dis_ap_l2_exp + dis_an_l2_exp)
            # dis =F.pairwise_distance(x,y, p=2)
            # score = torch.sigmoid(self.fc1(dis).squeeze(-1)).mean(0)
            score = dis_an_softmax
            # print(score)
        else:
            z = x * y
            z = self.fc(z).squeeze(-1)
            score = torch.sigmoid(z).mean(0)

        pred = []
        for i in range(score.size(0)):
            pred.append((score[i], i))
        pred.sort(key=lambda x: x[0], reverse=True)
        return pred