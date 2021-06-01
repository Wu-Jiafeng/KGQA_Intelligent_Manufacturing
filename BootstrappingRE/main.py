import os
import sys
import torch
import sklearn.metrics
import numpy as np
from sklearn.preprocessing import label_binarize
from BootstrappingRE.data import *
from BootstrappingRE.models import *
from torch import optim
from BootstrappingRE.config import parser
from tqdm import tqdm

eval_loss_txt = open("./output/eval_loss.txt", 'a', encoding="utf-8")
eval_txt = open("./output/eval_txt.txt", 'a', encoding="utf-8")
args = parser.parse_args()


def train_encoder():
    print("Mode: Training Encoder!")
    train_dataset = REDataset(args.train_data_file_path, args.word_vector_file_path, args.mode,
                              args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    val_dataset = REDataset(args.val_data_file_path, args.word_vector_file_path, args.mode,
                            args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    train_dataloader = REDataLoader(dataset=train_dataset, batch_size=args.encoder_batch_size,
                                    shuffle=args.shuffle, sampler=None)
    val_dataloader = REDataLoader(dataset=val_dataset, batch_size=args.encoder_batch_size)
    if args.encoder_type == "PCNN":
        encoder = PCNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                               args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
    else:
        encoder = CNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                              args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)

    model = Snowball(encoder, base_class=train_dataloader.dataset.rel_tot,
                     siamese_model=None, hidden_size=args.hidden_size)
    parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize, 1., weight_decay=1e-5)

    # start training
    print("Start Training Encoder!")
    model.train()

    start_iter = 0
    args.encoder_pretrained=os.path.join(args.ckpt_dir, args.encoder_name + ".pth.tar")
    if os.path.isfile(args.encoder_pretrained):
        checkpoint = torch.load(args.encoder_pretrained)
        print("Successfully loaded checkpoint '%s'" % args.encoder_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        start_iter = checkpoint['iter'] + 1
    else:
        print("Training encoder again! No checkpoint found!")

    if args.cuda: model = model.cuda()
    best_acc, global_step = 0, 0
    for epoch in range(start_iter, start_iter + args.encoder_training_epoch):
        epoch_step = 0
        iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0
        tbar = tqdm(train_dataloader, desc='Training Epoch ' + str(epoch))
        for batch_data in tbar:
            global_step += 1
            epoch_step += 1

            model.forward_base(batch_data)
            loss = model.loss() / args.grad_iter
            right = model.accuracy()
            loss.backward()

            # warmup
            cur_lr = args.encoder_training_lr
            if global_step < 300:
                cur_lr *= global_step / args.warmup_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

            if global_step % args.grad_iter == 0:
                optimizer.step()
                optimizer.zero_grad()

            iter_loss += loss
            iter_right += right
            iter_sample += 1

            tbar.set_postfix(loss=iter_loss / iter_sample, acc=iter_right / iter_sample)

        print('')
        acc = eval_encoder_one_epoch(model, val_dataloader, args.encoder_batch_size)
        print('')
        if acc > best_acc:
            print('Best checkpoint')
            if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)
            save_path = os.path.join(args.ckpt_dir, args.encoder_name + ".pth.tar")
            torch.save({'state_dict': model.state_dict()}, save_path)
            best_acc = acc

    print("\n####################\n")
    print("Finish training " + args.encoder_name)
    return


def eval_encoder_one_epoch(model, val_dataloader, batch_size=32):
    model.eval()
    iter_right = 0.0
    iter_sample = 0.0
    it = 0
    pred = []
    label = []
    tbar = tqdm(val_dataloader, desc='Validation')
    for batch_data in tbar:
        it += 1
        model.forward_base(batch_data)
        right = model.accuracy()
        iter_right += right
        iter_sample += 1
        tbar.set_postfix(acc=iter_right / iter_sample)
        pred.append(model._pred.cpu().detach().numpy())
        label.append(batch_data['label'].cpu().detach().numpy())
    model.train()
    pred = np.concatenate(pred)
    label = np.concatenate(label)
    pred = label_binarize(pred, classes=list(range(0, 13)) + list(range(14, val_dataloader.dataset.rel_tot)))
    label = label_binarize(label, classes=list(range(0, 13)) + list(range(14, val_dataloader.dataset.rel_tot)))
    micro_precision = sklearn.metrics.average_precision_score(pred, label, average='micro')
    micro_recall = sklearn.metrics.recall_score(pred, label, average='micro')
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    print('')
    print('micro precision: {}, micro recall: {}, micro f1: {}'.format(micro_precision, micro_recall, micro_f1))
    print('')
    return iter_right / iter_sample


def train_similarity_network():
    print("Mode: Training Similarity Network!")
    train_dataset = REDataset(args.train_data_file_path, args.word_vector_file_path, args.mode,
                              args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    val_dataset = REDataset(args.val_data_file_path, args.word_vector_file_path, args.mode,
                            args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    train_dataloader = REDataLoader(dataset=train_dataset, batch_size=args.encoder_batch_size,
                                    shuffle=args.shuffle, sampler=None)
    val_dataloader = REDataLoader(dataset=val_dataset, batch_size=args.encoder_batch_size)
    if args.encoder_type == "PCNN":
        encoder = PCNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                               args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
    else:
        encoder = CNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                              args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)

    if args.similarity_network_type == "Siamese":
        model = Siamese(encoder, hidden_size=args.hidden_size)
    else:
        model = Triplet(encoder, hidden_size=args.hidden_size)
    parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize, 1., weight_decay=1e-5)
    # start training
    print("Start Training " + args.similarity_network_type + " Similarity Network!")
    model.train()

    start_iter = 0
    args.similarity_network_pretrained = os.path.join(args.ckpt_dir, args.similarity_network_name + ".pth.tar")
    if os.path.isfile(args.similarity_network_pretrained):
        checkpoint = torch.load(args.similarity_network_pretrained)
        print("Successfully loaded checkpoint '%s'" % args.similarity_network_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        start_iter = checkpoint['iter'] + 1
    else:
        print("Training Similarity Network again! No checkpoint found!")

    if args.cuda: model = model.cuda()

    # Training
    global_step, best_prec = 0, 0

    not_best_count = 0  # Stop training after several epochs without improvement.
    iter_loss = 0.0
    iter_right, iter_sample = 0.0, 0.0
    iter_prec, iter_recall = 0.0, 0.0

    s_num_size = args.similarity_network_batch_size // args.similarity_network_multiclass_size

    tbar = tqdm(
        train_dataloader.next_multi_class(num_size=s_num_size, num_class=args.similarity_network_multiclass_size,
                                          total_iter=args.similarity_network_training_iter,
                                          start_iter=start_iter), desc='Training')
    for it, batch_data in tbar:
        global_step += 1

        model(batch_data, s_num_size, args.similarity_network_multiclass_size)

        loss = model._loss / args.grad_iter
        right = model._accuracy
        loss.backward()

        # warmup
        cur_lr = args.encoder_training_lr
        if global_step < 300:
            cur_lr *= global_step / args.warmup_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

        if global_step % args.grad_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        iter_loss += loss
        iter_right += right
        iter_sample += 1
        iter_prec += model._prec
        iter_recall += model._recall
        tbar.set_postfix(loss=iter_loss / iter_sample, acc=iter_right / iter_sample,
                         prec=iter_prec / iter_sample, recall=iter_recall / iter_sample)

        if it % args.similarity_network_training_val_iter == 0:
            eval_loss_txt.write(str(iter_loss / iter_sample) + '\n')
            iter_loss = 0.0
            iter_right, iter_sample = 0.0, 0.0
            iter_prec, iter_recall = 0.0, 0.0

        if (it + 1) % args.similarity_network_training_val_iter == 0:
            print('')
            prec = eval_similarity_network(model, val_dataloader, eval_iter=args.similarity_network_training_val_iter,
                                           threshold=0.5,
                                           s_num_size=s_num_size, s_num_class=args.similarity_network_multiclass_size)
            print('')
            if prec > best_prec:
                print('Best checkpoint')
                if not os.path.exists(args.ckpt_dir):
                    os.makedirs(args.ckpt_dir)
                save_path = os.path.join(args.ckpt_dir, args.similarity_network_name + ".pth.tar")
                torch.save({'state_dict': model.state_dict()}, save_path)
                best_prec = prec

    print("\n####################\n")
    print("Finish training " + args.similarity_network_name)


def eval_similarity_network(model, val_dataloader, s_num_size=10, s_num_class=10, eval_iter=2000, threshold=0.5):
    model.eval()
    iter_right, iter_sample = 0.0, 0.0
    iter_prec, iter_recall = 0.0, 0.0
    tbar = tqdm(val_dataloader.next_multi_class(num_size=s_num_size, num_class=s_num_class, total_iter=eval_iter,
                                                start_iter=0), desc='Validation')
    for it, batch_data in tbar:
        model(batch_data, s_num_size, s_num_class, threshold=threshold)
        iter_right += model._accuracy
        iter_prec += model._prec
        iter_recall += model._recall
        iter_sample += 1
        tbar.set_postfix(acc=iter_right / iter_sample, prec=iter_prec / iter_sample, recall=iter_recall / iter_sample)
    eval_txt.write(str(iter_right / iter_sample) + '\t' + str(iter_prec / iter_sample) + '\t' +
                   str(iter_recall / iter_sample) + '\n')
        # sys.stdout.write('[EVAL] step: {0:4} | acc: {1:3.2f}%, prec: {2:3.2f}%, recall: {3:3.2f}%'.format(it + 1,100 * iter_right / iter_sample,100 * iter_prec / iter_sample,100 * iter_recall / iter_sample) + '\r')
        # sys.stdout.flush()
    model.train()
    return iter_prec / iter_sample


def test_snowball():
    print("Mode: Testing Neural Snowball!")
    train_dataset = REDataset(args.train_data_file_path, args.word_vector_file_path, args.mode,
                              args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    val_dataset = REDataset(args.val_data_file_path, args.word_vector_file_path, args.mode,
                            args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    test_dataset = REDataset(args.test_data_file_path, args.word_vector_file_path, args.mode,
                             args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    train_dataloader = REDataLoader(dataset=train_dataset, batch_size=args.encoder_batch_size,
                                    shuffle=args.shuffle, sampler=None)
    val_dataloader = REDataLoader(dataset=val_dataset, batch_size=args.encoder_batch_size)
    test_dataloader = REDataLoader(dataset=test_dataset, batch_size=args.encoder_batch_size)

    if args.encoder_type == "PCNN":
        encoder1 = PCNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                                args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
        encoder2 = PCNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                                args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
    else:
        encoder1 = CNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                               args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
        encoder2 = CNN_Encoder(train_dataloader.dataset.word_vec_mat, args.max_length,
                               args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)

    if args.similarity_network_type == "Siamese":
        model2 = Siamese(encoder2, hidden_size=args.hidden_size)
    else:
        model2 = Triplet(encoder2, hidden_size=args.hidden_size)
    model = Snowball(encoder1, base_class=train_dataloader.dataset.rel_tot, neg_loader=train_dataloader,
                     siamese_model=model2, hidden_size=args.hidden_size, args=args)

    # load pretrain
    checkpoint = torch.load(os.path.join(args.ckpt_dir, args.encoder_name + ".pth.tar"))['state_dict']
    checkpoint2 = torch.load(os.path.join(args.ckpt_dir, args.similarity_network_name + ".pth.tar"))['state_dict']
    for key in checkpoint2:
        checkpoint['siamese_model.' + key] = checkpoint2[key]
    model.load_state_dict(checkpoint)

    if args.cuda: model = model.cuda()
    model.train()
    model_name = args.encoder_type+'_'+args.similarity_network_type
    res = eval_snowball(model, train_dataloader, val_dataloader, test_dataloader,
                        support_size=args.shot, query_size=50, eval_iter=args.snowball_eval_iter)
    res_file = open('exp_' + model_name + '_' + str(args.shot) + 'shot.txt', 'a')
    res_file.write(res + '\n')
    print('\n########## RESULT ##########')
    print(res)


def eval_baseline(model, train_dataloader, val_dataloader, test_dataloader, support_size=10, query_size=50,
                  eval_iter=2000,
                  unlabelled_size=50, s_num_size=10, s_num_class=10, ckpt=None, is_model2=False, threshold=0.5):
    '''
            model: a FewShotREModel instance
            B: Batch size
            N: Num of classes for each batch
            K: Num of instances for each class in the support set
            Q: Num of instances for each class in the query set
            eval_iter: Num of iterations
            ckpt: Checkpoint path. Set as None if using current model parameters.
            return: Accuracy
            '''
    print("")
    # model.eval()
    if ckpt is None:
        eval_dataloader = val_dataloader
    elif os.path.isfile(ckpt):
        checkpoint = torch.load(args.encoder_pretrained)
        print("Successfully loaded checkpoint '%s'" % args.encoder_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        eval_dataloader = test_dataloader
    else:
        print("No checkpoint found!")
        eval_dataloader = val_dataloader
    eval_distant_dataloader = test_dataloader

    iter_sample, iter_bright = 0.0, 0.0
    iter_bprec, iter_brecall = 0.0, 0.0
    snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32)]

    tbar = eval_dataloader.sample_for_eval(train_dataloader, support_size, query_size, eval_iter)
    for it, support_pos, support_neg, query, pos_class, neg_class in tbar:
        model.forward_baseline(support_pos, query, threshold=threshold)

        # support_pos, support_neg, query, pos_class = eval_dataloader.get_one_new_relation(
        #    train_dataloader, support_size, 10, query_size, query_class,
        #    use_train_neg=True, neg_train_loader=neg_train_loader)
        # model.forward_baseline(support_pos, support_neg, query, threshold=threshold)
        # model.forward(support_pos, support_neg, query, eval_distant_dataloader, pos_class, threshold=threshold)
        iter_bright += model._baseline_f1
        iter_bprec += model._baseline_prec
        iter_brecall += model._baseline_recall

        iter_sample += 1
        sys.stdout.write('[EVAL] step: {0:4} | [baseline] f1: {1:1.4f}, prec: {2:3.2f}%, rec: {3:3.2f}%'.format
                         (it + 1, iter_bright / iter_sample, 100 * iter_bprec / iter_sample,
                          100 * iter_brecall / iter_sample) + '\r')
        sys.stdout.flush()

    print('')
    return '[EVAL] step: {0:4} | [baseline] f1: {1:1.4f}, prec: {2:3.2f}%, rec: {3:3.2f}%' \
        .format(it + 1, iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample)


def eval_snowball(model, train_dataloader, val_dataloader, test_dataloader,support_size=10, query_size=50,
                  unlabelled_size=50, query_class=5, s_num_size=10, s_num_class=10,eval_iter=100, ckpt=None,
                  is_model2=False, threshold=0.5, query_train=False, query_val=True,eval_data=None):
    '''
    model: a FewShotREModel instance
    B: Batch size
    N: Num of classes for each batch
    K: Num of instances for each class in the support set
    Q: Num of instances for each class in the query set
    eval_iter: Num of iterations
    ckpt: Checkpoint path. Set as None if using current model parameters.
    return: Accuracy
    '''
    print("")
    model.eval()
    if ckpt is None:
        eval_dataloader = val_dataloader
    elif os.path.isfile(ckpt):
        checkpoint = torch.load(args.encoder_pretrained)
        print("Successfully loaded checkpoint '%s'" % args.encoder_pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        eval_dataloader = test_dataloader
    else:
        print("No checkpoint found!")
        eval_dataloader = val_dataloader
    eval_distant_dataloader = test_dataloader

    iter_sample = 0.0
    iter_right, iter_prec, iter_recall = 0.0, 0.0, 0.0
    iter_bright, iter_bprec, iter_brecall = 0.0, 0.0, 0.0
    snowball_metric = [np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32),
                       np.zeros([3], dtype=np.float32), np.zeros([3], dtype=np.float32)]

    tbar = eval_dataloader.sample_for_eval(train_dataloader, support_size, query_size, eval_iter,
                                           query_train=query_train, query_val=query_val)

    for it, support_pos, support_neg, query, pos_class, neg_class in tbar:
        # support_pos, support_neg, query, pos_class = eval_dataset.get_one_new_relation(self.train_data_loader,
        # support_size, 10, query_size, query_class, use_train_neg=True, neg_train_loader=self.neg_train_loader)
        model.forward_baseline(support_pos, query, threshold=threshold)
        model.forward(support_pos, support_neg, query, eval_distant_dataloader, pos_class, neg_class,
                      threshold=threshold)
        iter_bright += model._baseline_f1
        iter_bprec += model._baseline_prec
        iter_brecall += model._baseline_recall
        iter_right += model._f1
        iter_prec += model._prec
        iter_recall += model._recall
        iter_sample += 1
        sys.stdout.write(
            '[EVAL] step: {0:4} | f1: {1:1.4f}, prec: {2:3.2f}%, recall: {3:3.2f}% | [baseline] f1: {4:1.4f}, prec: {5:3.2f}%, rec: {6:3.2f}%'.format(
                it + 1, iter_right / iter_sample, 100 * iter_prec / iter_sample, 100 * iter_recall / iter_sample,
                iter_bright / iter_sample, 100 * iter_bprec / iter_sample, 100 * iter_brecall / iter_sample) + '\r')
        sys.stdout.flush()
        if args.eval:
            print("")
            print("[SNOWBALL ITER RESULT:]")
            for i in range(len(model._metric)):
                snowball_metric[i] += model._metric[i]
                print("iter {} : {}".format(i, snowball_metric[i] / iter_sample))
    res = "{} {} {} {} {} {}".format(iter_bright / iter_sample, iter_bprec / iter_sample,
                                     iter_brecall / iter_sample, iter_right / iter_sample, iter_prec / iter_sample,
                                     iter_recall / iter_sample)
    return res

def predict():
    print("Mode: Predicting Neural Snowball!")
    output=open("./output/output.txt","a",encoding="utf-8-sig")

    args.phase2_cl_th,args.phase2_siamese_th,args.phase1_siamese_th=0.7,0.3,0.5
    args.snowball_max_iter = 10
    args.phase2_add_num = 10
    train_dataset = REDataset(args.train_data_file_path, args.word_vector_file_path, args.mode,
                              args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    seed_dataset=REDataset(args.seed_data_file_path, args.word_vector_file_path, args.mode,
                           args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    distant_dataset=REDataset(args.unlabeled_data_file_path, args.word_vector_file_path, args.mode,
                              args.case_sensitive, args.max_length, args.cuda, args.reprocess)
    train_dataloader = REDataLoader(dataset=train_dataset, batch_size=args.encoder_batch_size,
                                    shuffle=args.shuffle, sampler=None)
    seed_dataloader = REDataLoader(dataset=seed_dataset, batch_size=args.encoder_batch_size,
                                    shuffle=args.shuffle, sampler=None)
    distant_dataloader = REDataLoader(dataset=distant_dataset, batch_size=args.encoder_batch_size,
                                      shuffle=args.shuffle, sampler=None)

    encoder1 = CNN_Encoder(seed_dataloader.dataset.word_vec_mat, args.max_length,
                            args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)
    encoder2 = CNN_Encoder(seed_dataloader.dataset.word_vec_mat, args.max_length,
                            args.word_embedding_dim, args.pos_embedding_dim, args.hidden_size, args.cuda)

    model2 = Triplet(encoder2, hidden_size=args.hidden_size)
    model = Snowball(encoder1, base_class=9, neg_loader=seed_dataloader,
                     siamese_model=model2, hidden_size=args.hidden_size, args=args)

    # load pretrain
    checkpoint = torch.load(os.path.join(args.ckpt_dir, args.encoder_name + ".pth.tar"))['state_dict']
    checkpoint2 = torch.load(os.path.join(args.ckpt_dir, args.similarity_network_name + ".pth.tar"))['state_dict']
    checkpoint['fc.weight']=checkpoint['fc.weight'][:9]
    checkpoint['fc.bias']=checkpoint['fc.bias'][:9]
    for key in checkpoint2:
        checkpoint['siamese_model.' + key] = checkpoint2[key]
    model.load_state_dict(checkpoint)
    if args.cuda: model = model.cuda()
    model.train()
    model.eval()
    model_name = args.encoder_type + '_' + args.similarity_network_type
    for pos_class in tqdm(seed_dataloader.dataset.rel2scope.keys(),desc="New Relation"):
        print(pos_class)
        for it, support_pos, support_neg, query, pos_class, neg_class in seed_dataloader.sample_for_eval(
                train_dataloader, 100, 50, 1,target_class=pos_class,query_train=False, query_val=True):
            support_pos_res=model.forward(support_pos, support_neg, query, distant_dataloader, pos_class, neg_class,
                                      threshold=0.5)
            for instance in support_pos_res:
                [sub,obj]=instance.split("#")
                output.write(sub+"\t"+pos_class+"\t"+obj+"\n")
            print(support_pos_res)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("menu:\n\t"
              "--m train_encoder --me cnn/pcnn\n\t"
              "--m train_sn --me cnn/pnn --ms sim/tri\n\t"
              "--m test --me cnn/pnn --ms sim/tri\n\t"
              "--m predict")
        exit()
    if sys.argv[2] == "train_encoder":
        if sys.argv[4]=="cnn":
            print("Encoder: CNN!\n")
            args.encoder_type = "CNN"
            args.encoder_name = "cnn_encoder_on_fewrel"
            train_encoder()
        else :
            print("Encoder: PCNN!\n")
            args.encoder_type = "PCNN"
            args.encoder_name = "pcnn_encoder_on_fewrel"
            train_encoder()
    elif sys.argv[2] == "train_sn" or sys.argv[2] == "test":
        if sys.argv[4]=="cnn" and sys.argv[6]=="sim":
            print("Encoder: CNN\tSimilarity Network: Simaese!\n")
            args.encoder_type = "CNN"
            args.encoder_name = "cnn_encoder_on_fewrel"
            args.similarity_network_type = "Simaese"
            args.similarity_network_name = "cnn_simaese_on_fewrel"
        elif sys.argv[4]=="cnn" and sys.argv[6]=="tri":
            print("Encoder: CNN\tSimilarity Network: Triplet!\n")
            args.encoder_type = "CNN"
            args.encoder_name = "cnn_encoder_on_fewrel"
            args.similarity_network_type = "Triplet"
            args.similarity_network_name = "cnn_triplet_on_fewrel"
        elif sys.argv[4]=="pcnn" and sys.argv[6]=="sim":
            print("Encoder: PCNN\tSimilarity Network: Simaese!\n")
            args.encoder_type = "PCNN"
            args.encoder_name = "pcnn_encoder_on_fewrel"
            args.similarity_network_type = "Simaese"
            args.similarity_network_name = "pcnn_simaese_on_fewrel"
        else:
            print("Encoder: PCNN\tSimilarity Network: Triplet!\n")
            args.encoder_type = "PCNN"
            args.encoder_name = "pcnn_encoder_on_fewrel"
            args.similarity_network_type = "Triplet"
            args.similarity_network_name = "pcnn_triplet_on_fewrel"

        if sys.argv[2] == "train_sn": train_similarity_network()
        else: test_snowball()
    elif sys.argv[2] == "predict":
        args.encoder_type = "CNN"
        args.encoder_name = "cnn_encoder_on_fewrel"
        args.similarity_network_type = "Triplet"
        args.similarity_network_name = "cnn_triplet_on_fewrel"
        predict()
