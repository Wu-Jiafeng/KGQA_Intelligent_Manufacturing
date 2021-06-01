import sys
import numpy as np
import random
sys.path.append('..')
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from BootstrappingRE.config import parser
from torch.nn import functional as F
import sklearn.metrics
import copy


class Snowball(nn.Module):

    def __init__(self, encoder, base_class, siamese_model, hidden_size=230, drop_rate=0.5, weight_table=None,
                 pre_rep=None, neg_loader=None, args=None):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.criterion = nn.BCELoss(size_average=True)
        self._loss = 0
        self._accuracy = 0
        self.parser = parser
        self.NA_label = None
        self.hidden_size = hidden_size
        self.base_class = base_class
        if encoder.model_type=="CNN": self.fc = nn.Linear(hidden_size, base_class)
        else: self.fc = nn.Linear(hidden_size*3, base_class)
        self.drop = nn.Dropout(drop_rate)
        self.siamese_model = siamese_model
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss(reduction="none")
        # self.criterion = nn.CrossEntropyLoss()
        self.weight_table = weight_table

        self.args = args

        self.pre_rep = pre_rep
        self.neg_loader = neg_loader

    def __loss__(self, logits, label):
        return self.criterion(logits.view(-1), label.view(-1))

    def __accuracy__(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        if self.NA_label is not None:
            pred = pred.view(-1).cpu().detach().numpy()
            label = label.view(-1).cpu().detach().numpy()
            return float(np.logical_and(label != self.NA_label, label == pred).sum()) / float((label != self.NA_label).sum() + 1)
        else:
            return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor)).item()

    def loss(self):
        return self._loss

    def accuracy(self):
        return self._accuracy

    def forward_base(self, data):
        batch_size = data['word'].size(0)
        x = self.encoder(data)  # (batch_size, hidden_size)
        x = self.drop(x)
        x = self.fc(x)  # (batch_size, base_class)

        x = torch.sigmoid(x)
        if self.weight_table is None:
            weight = 1.0
        else:
            weight = self.weight_table[data['label']].unsqueeze(1).expand(-1, self.base_class).contiguous().view(-1)
        label = torch.zeros((batch_size, self.base_class)).cuda()
        label.scatter_(1, data['label'].view(-1, 1), 1)  # (batch_size, base_class)
        loss_array = self.__loss__(x, label)
        self._loss = ((label.view(-1) + 1.0 / self.base_class) * weight * loss_array).mean() * self.base_class
        # self._loss = self.__loss__(x, data['label'])

        _, pred = x.max(-1)
        self._accuracy = self.__accuracy__(pred, data['label'])
        self._pred = pred

    def forward_baseline(self, support_pos, query, threshold=0.5):
        '''
        baseline model
        support_pos: positive support set
        support_neg: negative support set
        query: query set
        threshold: ins whose prob > threshold are predicted as positive
        '''

        # train
        self._train_finetune_init()
        # support_rep = self.encode(support, self.args.infer_batch_size)
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        # self._train_finetune(support_rep, support['label'])
        self._train_finetune(support_pos_rep)

        # test
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        self._baseline_accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1),
                                                      np.logical_and(query_prob < threshold,
                                                                     label == 0)).sum()) / float(query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            self._baseline_prec = 0
        else:
            self._baseline_prec = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float(
                (query_prob > threshold).sum())
        self._baseline_recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float(
            (label == 1).sum())
        if self._baseline_prec + self._baseline_recall == 0:
            self._baseline_f1 = 0
        else:
            self._baseline_f1 = float(2.0 * self._baseline_prec * self._baseline_recall) / float(
                self._baseline_prec + self._baseline_recall)
        self._baseline_auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            print('')
            sys.stdout.write(
                '[BASELINE EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format( \
                    self._baseline_accuracy * 100, self._baseline_prec * 100, self._baseline_recall * 100,
                    self._baseline_f1, self._baseline_auc))
            print('')

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def _train_finetune_init(self):
        # init variables and optimizer
        self.new_W = Variable(self.fc.weight.mean(0) / 1e3, requires_grad=True)
        self.new_bias = Variable(torch.zeros((1)), requires_grad=True)
        self.optimizer = optim.Adam([self.new_W, self.new_bias], self.args.finetune_lr,
                                    weight_decay=self.args.finetune_wd)
        self.new_W = self.new_W.cuda()
        self.new_bias = self.new_bias.cuda()

    def _train_finetune(self, data_repre, learning_rate=None, weight_decay=1e-5):
        '''
        train finetune classifier with given data
        data_repre: sentence representation (encoder's output)
        label: label
        '''

        self.train()

        optimizer = self.optimizer
        if learning_rate is not None:
            optimizer = optim.Adam([self.new_W, self.new_bias], learning_rate, weight_decay=weight_decay)

        # hyperparameters
        max_epoch = self.args.finetune_epoch
        batch_size = self.args.finetune_batch_size

        # dropout
        data_repre = self.drop(data_repre)

        # train
        if self.args.print_debug:
            print('')
        for epoch in range(max_epoch):
            max_iter = data_repre.size(0) // batch_size
            if data_repre.size(0) % batch_size != 0:
                max_iter += 1
            order = list(range(data_repre.size(0)))
            random.shuffle(order)
            for i in range(max_iter):
                x = data_repre[order[i * batch_size: min((i + 1) * batch_size, data_repre.size(0))]]
                # batch_label = label[order[i * batch_size : min((i + 1) * batch_size, data_repre.size(0))]]

                # neg sampling
                # ---------------------
                batch_label = torch.ones((x.size(0))).long().cuda()
                neg_size = int(x.size(0) * 1)
                neg = self.neg_loader.next_batch(neg_size)
                neg = self.encode(neg, self.args.infer_batch_size)
                x = torch.cat([x, neg], 0)
                batch_label = torch.cat([batch_label, torch.zeros((neg_size)).long().cuda()], 0)
                # ---------------------

                x = torch.matmul(x, self.new_W) + self.new_bias  # (batch_size, 1)
                x = torch.sigmoid(x)

                # iter_loss = self.__loss__(x, batch_label.float()).mean()
                weight = torch.ones(batch_label.size(0)).float().cuda()
                weight[batch_label == 0] = self.args.finetune_weight  # 1 / float(max_epoch)
                iter_loss = (self.__loss__(x, batch_label.float()) * weight).mean()

                optimizer.zero_grad()
                iter_loss.backward(retain_graph=True)
                optimizer.step()
                if self.args.print_debug:
                    sys.stdout.write('[snowball finetune] epoch {0:4} iter {1:4} | loss: {2:2.6f}'.format(epoch, i,
                                                                                                          iter_loss) + '\r')
                    sys.stdout.flush()
        self.eval()

    def _add_ins_to_data(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (list)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'].append(dataset_src['word'][ins_id])
        if 'posh' in dataset_src:
            dataset_dst['posh'].append(dataset_src['posh'][ins_id])
            dataset_dst['post'].append(dataset_src['post'][ins_id])
        dataset_dst['mask'].append(dataset_src['mask'][ins_id])
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'].append(dataset_src['id'][ins_id])
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'].append(label)

    def _add_ins_to_vdata(self, dataset_dst, dataset_src, ins_id, label=None):
        '''
        add one instance from dataset_src to dataset_dst (variable)
        dataset_dst: destination dataset
        dataset_src: source dataset
        ins_id: id of the instance
        '''
        dataset_dst['word'] = torch.cat([dataset_dst['word'], dataset_src['word'][ins_id].unsqueeze(0)], 0)
        if 'posh' in dataset_src:
            dataset_dst['posh'] = torch.cat([dataset_dst['posh'], dataset_src['posh'][ins_id].unsqueeze(0)], 0)
            dataset_dst['post'] = torch.cat([dataset_dst['post'], dataset_src['post'][ins_id].unsqueeze(0)], 0)
        dataset_dst['mask'] = torch.cat([dataset_dst['mask'], dataset_src['mask'][ins_id].unsqueeze(0)], 0)
        if 'id' in dataset_dst and 'id' in dataset_src:
            dataset_dst['id'] = torch.cat([dataset_dst['id'], dataset_src['id'][ins_id].unsqueeze(0)], 0)
        if 'entpair' in dataset_dst and 'entpair' in dataset_src:
            dataset_dst['entpair'].append(dataset_src['entpair'][ins_id])
        if 'label' in dataset_dst and label is not None:
            dataset_dst['label'] = torch.cat([dataset_dst['label'], torch.ones((1)).long().cuda()], 0)

    def _dataset_stack_and_cuda(self, dataset):
        '''
        stack the dataset to torch.Tensor and use cuda mode
        dataset: target dataset
        '''
        if (len(dataset['word']) == 0):
            return
        dataset['word'] = torch.stack(dataset['word'], 0).cuda()
        if 'posh' in dataset:
            dataset['posh'] = torch.stack(dataset['posh'], 0).cuda()
            dataset['post'] = torch.stack(dataset['post'], 0).cuda()
        dataset['mask'] = torch.stack(dataset['mask'], 0).cuda()
        dataset['id'] = torch.stack(dataset['id'], 0).cuda()

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

    def _infer(self, dataset, batch_size=0):
        '''
        get prob output of the finetune network with the input dataset
        dataset: input dataset
        return: prob output of the finetune network
        '''
        x = self.encode(dataset, batch_size=batch_size)
        x = torch.matmul(x, self.new_W) + self.new_bias  # (batch_size, 1)
        x = torch.sigmoid(x)
        return x.view(-1)

    def _forward_train(self, support_pos,support_neg, query, distant, threshold=0.5):
        '''
        snowball process (train)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set
        distant: distant data loader
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_phase1: distant ins with prob > th_for_phase1 will be added to extended support set at phase1
        threshold_for_phase2: distant ins with prob > th_for_phase2 will be added to extended support set at phase2
        '''

        # hyperparameters
        snowball_max_iter = self.args.snowball_max_iter
        sys.stdout.flush()
        candidate_num_class = self.base_class
        candidate_num_ins_per_class = 100

        sort_num1 = self.args.phase1_add_num
        sort_num2 = self.args.phase2_add_num
        sort_threshold1 = self.args.phase1_siamese_th
        sort_threshold2 = self.args.phase2_siamese_th
        sort_ori_threshold = self.args.phase2_cl_th

        # get neg representations with sentence encoder
        # support_neg_rep = self.encode(support_neg, batch_size=self.args.infer_batch_size)

        # init
        self._train_finetune_init()
        # support_rep = self.encode(support, self.args.infer_batch_size)
        support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
        #support_neg_rep = self.encode(support_neg, self.args.infer_batch_size)
        # self._train_finetune(support_rep, support['label'])
        self._train_finetune(support_pos_rep)

        self._metric = []

        # copy
        #original_support_pos = copy.deepcopy(support_pos)
        #original_support_neg = copy.deepcopy(support_neg)

        ## get negative set
        #del support_neg['entpair']

        # snowball
        exist_id = {}
        if self.args.print_debug:
            print('\n-------------------------------------------------------')
        for snowball_iter in range(snowball_max_iter):
            if self.args.print_debug:
                print('###### snowball iter ' + str(snowball_iter))
            # phase 1: expand positive support set from distant dataset (with same entity pairs)

            ## get all entpairs and their ins in positive support set
            old_support_pos_label = support_pos['label'] + 0
            entpair_support= {}
            entpair_distant = {}
            for i in range(len(support_pos['id'])):  # only positive support
                entpair = support_pos['entpair'][i]
                exist_id[support_pos['id'][i]] = 1
                if entpair not in entpair_support:
                    if 'posh' in support_pos:
                        entpair_support[entpair] = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': []}
                    else:
                        entpair_support[entpair] = {'word': [], 'mask': [], 'id': []}
                self._add_ins_to_data(entpair_support[entpair], support_pos, i)

            ## pick all ins with the same entpairs in distant data and choose with siamese network
            self._phase1_add_num = 0  # total number of snowball instances
            self._phase1_total = 0
            for entpair in entpair_support:
                raw = distant.get_same_entpair_ins(entpair)  # ins with the same entpair
                if raw is None:
                    continue
                if 'posh' in support_pos:
                    entpair_distant[entpair] = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': [], 'entpair': []}
                else:
                    entpair_distant[entpair] = {'word': [], 'mask': [], 'id': [], 'entpair': []}
                for i in range(raw['word'].size(0)):
                    if raw['id'][i] not in exist_id:  # don't pick sentences already in the support set
                        self._add_ins_to_data(entpair_distant[entpair], raw, i)
                self._dataset_stack_and_cuda(entpair_support[entpair])
                self._dataset_stack_and_cuda(entpair_distant[entpair])
                if len(entpair_support[entpair]['word']) == 0 or len(entpair_distant[entpair]['word']) == 0:
                    continue

                if  self.siamese_model.model_type=="Triplet":
                    pick_or_not = self.siamese_model.forward_infer_sort(entpair_support[entpair],
                                                                        entpair_distant[entpair],support_neg,
                                                                        batch_size=self.args.infer_batch_size)
                else:
                    pick_or_not = self.siamese_model.forward_infer_sort(entpair_support[entpair],
                                                                        entpair_distant[entpair],
                                                                        batch_size=self.args.infer_batch_size)

                # pick_or_not = self.siamese_model.forward_infer_sort(original_support_pos, entpair_distant[entpair], threshold=threshold_for_phase1)
                # pick_or_not = self._infer(entpair_distant[entpair]) > threshold

                # -- method B: use sort --
                for i in range(min(len(pick_or_not), sort_num1)):
                    if pick_or_not[i][0] > sort_threshold1:
                        iid = pick_or_not[i][1]
                        self._add_ins_to_vdata(support_pos, entpair_distant[entpair], iid, label=1)
                        exist_id[entpair_distant[entpair]['id'][iid]] = 1
                        self._phase1_add_num += 1
                self._phase1_total += entpair_distant[entpair]['word'].size(0)

            support_pos_rep = self.encode(support_pos, batch_size=self.args.infer_batch_size)
            # support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            # support_label = torch.cat([support_pos['label'], support_neg['label']], 0)

            ## finetune
            # print("Fine-tune Init")
            self._train_finetune_init()
            self._train_finetune(support_pos_rep)
            if self.args.eval:
                self._forward_eval_binary(query, threshold)
            # self._metric.append(np.array([self._f1, self._prec, self._recall]))
            if self.args.print_debug:
                print('\nphase1 add {} ins / {}'.format(self._phase1_add_num, self._phase1_total))

            # phase 2: use the new classifier to pick more extended support ins
            self._phase2_add_num = 0
            candidate = distant.get_random_candidate(self.pos_class, candidate_num_class, candidate_num_ins_per_class)

            ## -- method 1: directly use the classifier --
            candidate_prob = self._infer(candidate, batch_size=self.args.infer_batch_size)
            ## -- method 2: use siamese network --

            if self.siamese_model.model_type=="Triplet":
                pick_or_not = self.siamese_model.forward_infer_sort(support_pos, candidate,support_neg,
                                                                    batch_size=self.args.infer_batch_size)
            else:
                pick_or_not = self.siamese_model.forward_infer_sort(support_pos, candidate,
                                                                    batch_size=self.args.infer_batch_size)

            ## -- method A: use threshold ---- method B: use sort --
            self._phase2_total = candidate['word'].size(0)
            for i in range(min(len(candidate_prob), sort_num2)):
                iid = pick_or_not[i][1]
                if (pick_or_not[i][0] > sort_threshold2) and (candidate_prob[iid] > sort_ori_threshold) and not (
                        candidate['id'][iid] in exist_id):
                    exist_id[candidate['id'][iid]] = 1
                    self._phase2_add_num += 1
                    self._add_ins_to_vdata(support_pos, candidate, iid, label=1)

            ## build new support set
            support_pos_rep = self.encode(support_pos, self.args.infer_batch_size)
            # support_rep = torch.cat([support_pos_rep, support_neg_rep], 0)
            # support_label = torch.cat([support_pos['label'], support_neg['label']], 0)

            ## finetune
            # print("Fine-tune Init")
            self._train_finetune_init()
            self._train_finetune(support_pos_rep)
            if self.args.eval:
                self._forward_eval_binary(query, threshold)
                self._metric.append(np.array([self._f1, self._prec, self._recall]))
            if self.args.print_debug:
                print('\nphase2 add {} ins / {}'.format(self._phase2_add_num, self._phase2_total))

        self._forward_eval_binary(query, threshold)
        if self.args.print_debug:
            print('\nphase2 add {} ins / {}'.format(self._phase2_add_num, self._phase2_total))

        return set(support_pos["entpair"])

    def _forward_eval_binary(self, query, threshold=0.5):
        '''
        snowball process (eval)
        query: query set (raw data)
        threshold: ins with prob > threshold will be classified as positive
        return (accuracy at threshold, precision at threshold, recall at threshold, f1 at threshold, auc),
        '''
        query_prob = self._infer(query, batch_size=self.args.infer_batch_size).cpu().detach().numpy()
        label = query['label'].cpu().detach().numpy()
        accuracy = float(np.logical_or(np.logical_and(query_prob > threshold, label == 1),
                                       np.logical_and(query_prob < threshold, label == 0)).sum()) / float(
            query_prob.shape[0])
        if (query_prob > threshold).sum() == 0:
            precision = 0
        else:
            precision = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float(
                (query_prob > threshold).sum())
        recall = float(np.logical_and(query_prob > threshold, label == 1).sum()) / float((label == 1).sum())
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = float(2.0 * precision * recall) / float(precision + recall)
        auc = sklearn.metrics.roc_auc_score(label, query_prob)
        if self.args.print_debug:
            print('')
            sys.stdout.write(
                '[EVAL] acc: {0:2.2f}%, prec: {1:2.2f}%, rec: {2:2.2f}%, f1: {3:1.3f}, auc: {4:1.3f}'.format( \
                    accuracy * 100, precision * 100, recall * 100, f1, auc) + '\r')
            sys.stdout.flush()
        self._accuracy = accuracy
        self._prec = precision
        self._recall = recall
        self._f1 = f1
        return (accuracy, precision, recall, f1, auc)

    def forward(self, support_pos,support_neg, query, distant, pos_class,neg_class,threshold=0.5, threshold_for_snowball=0.5):
        '''
        snowball process (train + eval)
        support_pos: support set (positive, raw data)
        support_neg: support set (negative, raw data)
        query: query set (raw data)
        distant: distant data loader
        pos_class: positive relation (name)
        neg_class: negative relation (name)
        threshold: ins with prob > threshold will be classified as positive
        threshold_for_snowball: distant ins with prob > th_for_snowball will be added to extended support set
        '''
        self.pos_class = pos_class

        return self._forward_train(support_pos,support_neg, query, distant, threshold=threshold)