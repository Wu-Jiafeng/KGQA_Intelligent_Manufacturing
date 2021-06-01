import os
import gc
import random
import struct
import torch
import numpy as np
import ujson as json
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from BootstrappingRE.config import parser
#from data.gen_id import gen_id

args = parser.parse_args()

class REDistantDataset(Dataset):
    def __init__(self,file_name,word_vec_file_name,mode="Train",
                 case_sensitive=True, max_length=40, cuda=True,reprocess=False):
        self.file_name=file_name
        self.word_vec_file_name=word_vec_file_name
        self.mode=mode
        self.case_sensitive=case_sensitive
        self.max_length=max_length
        self.cuda=cuda

        # check if the data has been preprocessed
        processed_data_dir=args.processed_data_dir
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix='.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        if not os.path.exists(processed_data_dir):
            print("The processed data directory does not exist! Time to create!")
            os.makedirs(processed_data_dir)

        self.uid_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_uid.npy')
        self.word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        self.posh_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_posh.npy')
        self.post_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_post.npy')
        self.mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        self.length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        self.label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        self.entpair_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair.npy')
        self.rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        self.word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        self.word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        self.rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        self.entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        if not os.path.exists(self.word_vec_mat_file_name) or \
                not os.path.exists(self.word2id_file_name):
            print("Preprocessed word vector files do not exist. Now preprocess the word vector data...")
            self.preprocess_word_vector_data()
        print("Pre-processed word vector files exist. Loading them...")
        self.word_vec_mat = np.load(self.word_vec_mat_file_name)
        self.word2id = json.load(open(self.word2id_file_name,"r",encoding="utf-8"))

        if not os.path.exists(self.uid_npy_file_name):
            print("Preprocessed uid files do not exist. You must excute gen_id function first!")
            #gen_id() # 以后可以加入到self中

        if reprocess or not os.path.exists(self.word_npy_file_name) or \
                not os.path.exists(self.posh_npy_file_name) or \
                not os.path.exists(self.post_npy_file_name) or \
                not os.path.exists(self.mask_npy_file_name) or \
                not os.path.exists(self.length_npy_file_name) or \
                not os.path.exists(self.label_npy_file_name) or \
                not os.path.exists(self.entpair_npy_file_name) or \
                not os.path.exists(self.rel2scope_file_name) or \
                not os.path.exists(self.rel2id_file_name) or \
                not os.path.exists(self.entpair2scope_file_name):
            print("Preprocessed original files do not exist. Now preprocess the original data...")
            self.preprocess_original_data()

        # loading preprocessed data...
        print("Pre-processed original files exist. Loading them...")
        self.uid=np.load(self.uid_npy_file_name)
        self.data_word = np.load(self.word_npy_file_name)
        self.data_posh = np.load(self.posh_npy_file_name)
        self.data_post = np.load(self.post_npy_file_name)
        self.data_mask = np.load(self.mask_npy_file_name)
        self.data_length = np.load(self.length_npy_file_name)
        self.data_label = np.load(self.label_npy_file_name)
        self.data_entpair = np.load(self.entpair_npy_file_name)
        self.rel2scope = json.load(open(self.rel2scope_file_name,"r",encoding="utf-8"))
        self.rel2id = json.load(open(self.rel2id_file_name,"r",encoding="utf-8"))
        self.entpair2scope = json.load(open(self.entpair2scope_file_name,"r",encoding="utf-8"))
        self.instance_tot = self.data_word.shape[0]
        self.rel_tot = len(self.rel2id)
        print("Finish loading")

        self.id2rel = {}
        for rel in self.rel2id:
            self.id2rel[self.rel2id[rel]] = rel
        self.index_list = list(range(self.instance_tot))
        self.current = 0

    def __len__(self): return self.instance_tot

    def __getitem__(self,index):
        if index>=self.__len__():
            print("Index Error! Please check the variable index!")
            return None
        #if self.current >= self.__len__():
        #    if self.shuffle: random.shuffle(self.index_list)
        #    self.current=0
        #    return None
        instance={}
        instance['word'] = self.data_word[index]
        instance['posh'] = self.data_posh[index]
        instance['post'] = self.data_post[index]
        instance['mask'] = self.data_mask[index]
        instance['label'] = self.data_label[index]
        instance['id'] = self.uid[index]

        return instance

    def preprocess_word_vector_data(self):
        # loading the original word vector data and Pre-process word vec
        print("Loading word vector file...")
        ori_word_vec = open(self.word_vec_file_name, "rb")
        record_struct = struct.Struct("ii")
        self.word_vec_tot, self.word_vec_dim = struct.unpack("ii", ori_word_vec.read(record_struct.size))
        print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
        print("Building word vector matrix and mapping...")
        UNK = self.word_vec_tot
        BLANK = self.word_vec_tot + 1
        self.word2id = {}
        self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
        for cur_id in tqdm(range(self.word_vec_tot)):
            record_struct = struct.Struct("i")
            word_len = struct.unpack("i", ori_word_vec.read(record_struct.size))[0]
            word = ori_word_vec.read(word_len).decode("utf-8")
            if not self.case_sensitive:
                word = word.lower()
            record_struct = struct.Struct("f" * self.word_vec_dim)
            vector = struct.unpack("f" * self.word_vec_dim, ori_word_vec.read(record_struct.size))
            self.word2id[word] = cur_id
            self.word_vec_mat[cur_id, :] = vector
            self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
        self.word2id['UNK'] = UNK
        self.word2id['BLANK'] = BLANK
        print("Finish loading and building! Now store files!")
        np.save(self.word_vec_mat_file_name, self.word_vec_mat)
        json.dump(self.word2id, open(self.word2id_file_name, 'w',encoding="utf-8"),ensure_ascii=False)
        print("Finish storing word vector files!")

    def preprocess_original_data(self):
        #loading the original data...
        print("Loading the original data file...")
        self.ori_data = json.load(open(self.file_name, "r",encoding="utf-8"))
        print("Finish loading")

        # Eliminate case sensitive
        if not self.case_sensitive:
            print("Elimiating case sensitive problem...")
            for key,instances in self.ori_data.items():
                for instance in self.ori_data[instances]:
                    instance['tokens']=[token.lower() for token in instance['tokens']]
            print("Finish eliminating")

        # Pre-process data
        print("Pre-processing data...")
        self.instance_tot = 0
        for relation in self.ori_data:
            self.instance_tot += len(self.ori_data[relation])
        self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_posh = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_post = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
        self.data_label = np.zeros((self.instance_tot), dtype=np.int32)
        self.data_entpair = []
        self.rel2scope = {}  # left close right open
        self.entpair2scope = {}
        self.rel2id = {}
        self.rel_tot = len(self.ori_data)
        i = 0
        for cur_id,(relation,instance_list) in tqdm(enumerate(self.ori_data.items())):
            self.rel2scope[relation] = [i, i]
            self.rel2id[relation] = cur_id
            for ins in instance_list:
                tokens = ins['token']
                head,tail = ins['h'][0],ins['t'][0]
                posh,post = ins['h'][2][0][0],ins['t'][2][0][0]
                data_word2id = self.data_word[i]
                entpair = head + '#' + tail
                self.data_entpair.append(entpair) # record entity pair per instance

                # map the tokens to word2id
                for j, token in enumerate(tokens):
                    if j < self.max_length:
                        if token in self.word2id: data_word2id[j] = self.word2id[token]
                        else: data_word2id[j] = self.word2id['UNK']
                for j in range(len(tokens), self.max_length): data_word2id[j] = self.word2id['BLANK']

                # get the relative position (to head or tail) of tokens and divide them into 4 parts:
                # ---part1----entity1----part2----entity2-----part3-----maxlength-----part0-----
                self.data_label[i] = self.rel2id[relation]
                self.data_length[i] = min(len(tokens),self.max_length)
                posh,post=min(posh,self.max_length-1),min(post,self.max_length-1)
                pos_min,pos_max = min(posh, post),max(posh, post)
                for j in range(self.max_length):
                    self.data_posh[i][j] = j - posh + self.max_length
                    self.data_post[i][j] = j - post + self.max_length
                    if j >= self.data_length[i]: self.data_mask[i][j] = 0
                    elif j <= pos_min: self.data_mask[i][j] = 1
                    elif j <= pos_max: self.data_mask[i][j] = 2
                    else: self.data_mask[i][j] = 3

                # get entity pair instance set scope
                if not entpair in self.entpair2scope: self.entpair2scope[entpair] = [i]
                else: self.entpair2scope[entpair].append(i)
                i += 1
            # get relation instance set scope
            self.rel2scope[relation][1] = i

        # store files
        print("Finish pre-processing! Now Store files!")
        self.data_entpair = np.array(self.data_entpair)
        np.save(self.word_npy_file_name, self.data_word)
        np.save(self.posh_npy_file_name, self.data_posh)
        np.save(self.post_npy_file_name, self.data_post)
        np.save(self.mask_npy_file_name, self.data_mask)
        np.save(self.length_npy_file_name, self.data_length)
        np.save(self.label_npy_file_name, self.data_label)
        np.save(self.entpair_npy_file_name, self.data_entpair)
        json.dump(self.rel2scope, open(self.rel2scope_file_name,"w",encoding="utf8"),ensure_ascii=False)
        json.dump(self.rel2id, open(self.rel2id_file_name,"w",encoding="utf8"),ensure_ascii=False)
        json.dump(self.entpair2scope, open(self.entpair2scope_file_name,"w",encoding="utf8"),ensure_ascii=False)
        print("Finish storing")


class REDataset(Dataset):
    def __init__(self,file_name,word_vec_file_name,mode="Train",
                 case_sensitive=True, max_length=40, cuda=True,reprocess=False):
        self.file_name=file_name
        self.word_vec_file_name=word_vec_file_name
        self.mode=mode
        self.case_sensitive=case_sensitive
        self.max_length=max_length
        self.cuda=cuda

        # check if the data has been preprocessed
        processed_data_dir=args.processed_data_dir
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix='.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        if not os.path.exists(processed_data_dir):
            print("The processed data directory does not exist! Time to create!")
            os.makedirs(processed_data_dir)

        self.uid_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_uid.npy')
        self.word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        self.posh_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_posh.npy')
        self.post_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_post.npy')
        self.mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        self.length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        self.label_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_label.npy')
        self.entpair_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair.npy')
        self.rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        self.word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        self.word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        self.rel2id_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2id.json')
        self.entpair2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_entpair2scope.json')
        if not os.path.exists(self.word_vec_mat_file_name) or \
                not os.path.exists(self.word2id_file_name):
            print("Preprocessed word vector files do not exist. Now preprocess the word vector data...")
            self.preprocess_word_vector_data()
        print("Pre-processed word vector files exist. Loading them...")
        self.word_vec_mat = np.load(self.word_vec_mat_file_name)
        self.word2id = json.load(open(self.word2id_file_name,"r",encoding="utf-8"))

        if not os.path.exists(self.uid_npy_file_name):
            print("Preprocessed uid files do not exist. You must excute gen_id function first!")
            #gen_id() # 以后可以加入到self中

        if reprocess or not os.path.exists(self.word_npy_file_name) or \
                not os.path.exists(self.posh_npy_file_name) or \
                not os.path.exists(self.post_npy_file_name) or \
                not os.path.exists(self.mask_npy_file_name) or \
                not os.path.exists(self.length_npy_file_name) or \
                not os.path.exists(self.label_npy_file_name) or \
                not os.path.exists(self.entpair_npy_file_name) or \
                not os.path.exists(self.rel2scope_file_name) or \
                not os.path.exists(self.rel2id_file_name) or \
                not os.path.exists(self.entpair2scope_file_name):
            print("Preprocessed original files do not exist. Now preprocess the original data...")
            self.preprocess_original_data()

        # loading preprocessed data...
        print("Pre-processed original files exist. Loading them...")
        self.uid=np.load(self.uid_npy_file_name)
        self.data_word = np.load(self.word_npy_file_name)
        self.data_posh = np.load(self.posh_npy_file_name)
        self.data_post = np.load(self.post_npy_file_name)
        self.data_mask = np.load(self.mask_npy_file_name)
        self.data_length = np.load(self.length_npy_file_name)
        self.data_label = np.load(self.label_npy_file_name)
        self.data_entpair = np.load(self.entpair_npy_file_name)
        self.rel2scope = json.load(open(self.rel2scope_file_name,"r",encoding="utf-8"))
        self.rel2id = json.load(open(self.rel2id_file_name,"r",encoding="utf-8"))
        self.entpair2scope = json.load(open(self.entpair2scope_file_name,"r",encoding="utf-8"))
        self.instance_tot = self.data_word.shape[0]
        self.rel_tot = len(self.rel2id)
        print("Finish loading")

        self.id2rel = {}
        for rel in self.rel2id:
            self.id2rel[self.rel2id[rel]] = rel
        self.index_list = list(range(self.instance_tot))
        self.current = 0

    def __len__(self): return self.instance_tot

    def __getitem__(self,index):
        if index>=self.__len__():
            print("Index Error! Please check the variable index!")
            return None
        #if self.current >= self.__len__():
        #    if self.shuffle: random.shuffle(self.index_list)
        #    self.current=0
        #    return None
        instance={}
        instance['word'] = self.data_word[index]
        instance['posh'] = self.data_posh[index]
        instance['post'] = self.data_post[index]
        instance['mask'] = self.data_mask[index]
        instance['label'] = self.data_label[index]
        instance['id'] = self.uid[index]

        return instance

    def preprocess_word_vector_data(self):
        # loading the original word vector data and Pre-process word vec
        print("Loading word vector file...")
        ori_word_vec = open(self.word_vec_file_name, "rb")
        record_struct = struct.Struct("ii")
        self.word_vec_tot, self.word_vec_dim = struct.unpack("ii", ori_word_vec.read(record_struct.size))
        print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
        print("Building word vector matrix and mapping...")
        UNK = self.word_vec_tot
        BLANK = self.word_vec_tot + 1
        self.word2id = {}
        self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
        for cur_id in tqdm(range(self.word_vec_tot)):
            record_struct = struct.Struct("i")
            word_len = struct.unpack("i", ori_word_vec.read(record_struct.size))[0]
            word = ori_word_vec.read(word_len).decode("utf-8")
            if not self.case_sensitive:
                word = word.lower()
            record_struct = struct.Struct("f" * self.word_vec_dim)
            vector = struct.unpack("f" * self.word_vec_dim, ori_word_vec.read(record_struct.size))
            self.word2id[word] = cur_id
            self.word_vec_mat[cur_id, :] = vector
            self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
        self.word2id['UNK'] = UNK
        self.word2id['BLANK'] = BLANK
        print("Finish loading and building! Now store files!")
        np.save(self.word_vec_mat_file_name, self.word_vec_mat)
        json.dump(self.word2id, open(self.word2id_file_name, 'w',encoding="utf-8"),ensure_ascii=False)
        print("Finish storing word vector files!")

    def preprocess_original_data(self):
        #loading the original data...
        print("Loading the original data file...")
        self.ori_data = json.load(open(self.file_name, "r",encoding="utf-8"))
        print("Finish loading")

        # Eliminate case sensitive
        if not self.case_sensitive:
            print("Elimiating case sensitive problem...")
            for key,instances in self.ori_data.items():
                for instance in self.ori_data[instances]:
                    instance['tokens']=[token.lower() for token in instance['tokens']]
            print("Finish eliminating")

        # Pre-process data
        print("Pre-processing data...")
        self.instance_tot = 0
        for relation in self.ori_data:
            self.instance_tot += len(self.ori_data[relation])
        self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_posh = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_post = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
        self.data_label = np.zeros((self.instance_tot), dtype=np.int32)
        self.data_entpair = []
        self.rel2scope = {}  # left close right open
        self.entpair2scope = {}
        self.rel2id = {}
        self.rel_tot = len(self.ori_data)
        i = 0
        for cur_id,(relation,instance_list) in tqdm(enumerate(self.ori_data.items())):
            self.rel2scope[relation] = [i, i]
            self.rel2id[relation] = cur_id
            for ins in instance_list:
                tokens = ins['token']
                head,tail = ins['h'][0],ins['t'][0]
                posh,post = ins['h'][2][0][0],ins['t'][2][0][0]
                data_word2id = self.data_word[i]
                entpair = head + '#' + tail
                self.data_entpair.append(entpair) # record entity pair per instance

                # map the tokens to word2id
                for j, token in enumerate(tokens):
                    if j < self.max_length:
                        if token in self.word2id: data_word2id[j] = self.word2id[token]
                        else: data_word2id[j] = self.word2id['UNK']
                for j in range(len(tokens), self.max_length): data_word2id[j] = self.word2id['BLANK']

                # get the relative position (to head or tail) of tokens and divide them into 4 parts:
                # ---part1----entity1----part2----entity2-----part3-----maxlength-----part0-----
                self.data_label[i] = self.rel2id[relation]
                self.data_length[i] = min(len(tokens),self.max_length)
                posh,post=min(posh,self.max_length-1),min(post,self.max_length-1)
                pos_min,pos_max = min(posh, post),max(posh, post)
                for j in range(self.max_length):
                    self.data_posh[i][j] = j - posh + self.max_length
                    self.data_post[i][j] = j - post + self.max_length
                    if j >= self.data_length[i]: self.data_mask[i][j] = 0
                    elif j <= pos_min: self.data_mask[i][j] = 1
                    elif j <= pos_max: self.data_mask[i][j] = 2
                    else: self.data_mask[i][j] = 3

                # get entity pair instance set scope
                if not entpair in self.entpair2scope: self.entpair2scope[entpair] = [i]
                else: self.entpair2scope[entpair].append(i)
                i += 1
            # get relation instance set scope
            self.rel2scope[relation][1] = i

        # store files
        print("Finish pre-processing! Now Store files!")
        self.data_entpair = np.array(self.data_entpair)
        np.save(self.word_npy_file_name, self.data_word)
        np.save(self.posh_npy_file_name, self.data_posh)
        np.save(self.post_npy_file_name, self.data_post)
        np.save(self.mask_npy_file_name, self.data_mask)
        np.save(self.length_npy_file_name, self.data_length)
        np.save(self.label_npy_file_name, self.data_label)
        np.save(self.entpair_npy_file_name, self.data_entpair)
        json.dump(self.rel2scope, open(self.rel2scope_file_name,"w",encoding="utf8"),ensure_ascii=False)
        json.dump(self.rel2id, open(self.rel2id_file_name,"w",encoding="utf8"),ensure_ascii=False)
        json.dump(self.entpair2scope, open(self.entpair2scope_file_name,"w",encoding="utf8"),ensure_ascii=False)
        print("Finish storing")


def RE_collate_fn(data):
    batch={'word':[],'posh':[],'post':[],'mask':[],'label':[],'id':[]}
    for instance in data:
        for key in instance: batch[key].append(instance[key])

    for key in batch:
        batch[key]=np.array(batch[key])
        if args.cuda: batch[key]=Variable(torch.from_numpy(batch[key]).long()).cuda()
        else: batch[key]=Variable(torch.from_numpy(batch[key]).long())
    return batch

class REDataLoader(DataLoader):
    def __init__(self,dataset,batch_size,shuffle=False,sampler=None):
        DataLoader.__init__(self,dataset=dataset,batch_size=batch_size,
                            shuffle=shuffle,sampler=sampler,collate_fn=RE_collate_fn)

    def next_multi_class(self, num_size, num_class,total_iter,start_iter):
        '''
        num_size: The num of instances for ONE class. The total size is num_size * num_classes.
        num_class: The num of classes (include the positive class).
        '''

        for step in range(start_iter,start_iter+total_iter):
            target_classes = random.sample(self.dataset.rel2scope.keys(), num_class)
            batch = {'word': [], 'posh': [], 'post': [], 'mask': []}
            for i, class_name in enumerate(target_classes):
                scope = self.dataset.rel2scope[class_name]
                indices = np.random.choice(list(range(scope[0], scope[1])), num_size, True)
                batch['word'].append(self.dataset.data_word[indices])
                batch['posh'].append(self.dataset.data_posh[indices])
                batch['post'].append(self.dataset.data_post[indices])
                batch['mask'].append(self.dataset.data_mask[indices])

            for key in batch:
                batch[key]=np.concatenate(batch[key], 0)
                batch[key]=Variable(torch.from_numpy(batch[key]).long())
                if args.cuda: batch[key] = batch[key].cuda() # To cuda

            yield step,batch

    def next_batch(self, batch_size):
        batch = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': []}
        if self.dataset.current + batch_size > len(self.dataset.index_list):
            self.dataset.index_list = list(range(self.dataset.instance_tot))
            self.dataset.current = 0
        current_index = self.dataset.index_list[self.dataset.current:self.dataset.current+batch_size]
        self.dataset.current += batch_size

        batch['word'] = Variable(torch.from_numpy(self.dataset.data_word[current_index]).long())
        batch['posh'] = Variable(torch.from_numpy(self.dataset.data_posh[current_index]).long())
        batch['post'] = Variable(torch.from_numpy(self.dataset.data_post[current_index]).long())
        batch['mask'] = Variable(torch.from_numpy(self.dataset.data_mask[current_index]).long())
        batch['id'] = Variable(torch.from_numpy(self.dataset.uid[current_index]).long())
        batch['label']= Variable(torch.from_numpy(self.dataset.data_label[current_index]).long())

        # To cuda
        if args.cuda:
            for key in batch: batch[key] = batch[key].cuda()

        return batch

    def get_same_entpair_ins(self, entpair):
        '''
        return instances with the same entpair
        entpair: a string with the format '$head_entity#$tail_entity'
        '''
        if not entpair in self.dataset.entpair2scope:
            return None
        scope = self.dataset.entpair2scope[entpair]
        batch = {}
        batch['word'] = Variable(torch.from_numpy(self.dataset.data_word[scope]).long())
        batch['posh'] = Variable(torch.from_numpy(self.dataset.data_posh[scope]).long())
        batch['post'] = Variable(torch.from_numpy(self.dataset.data_post[scope]).long())
        batch['mask'] = Variable(torch.from_numpy(self.dataset.data_mask[scope]).long())
        batch['label']= Variable(torch.from_numpy(self.dataset.data_label[scope]).long())
        batch['id'] = Variable(torch.from_numpy(self.dataset.uid[scope]).long())
        batch['entpair'] = [entpair] * len(scope)

        # To cuda
        if args.cuda:
            for key in ['word', 'posh', 'post', 'mask', 'id']: batch[key] = batch[key].cuda()

        return batch

    def get_random_candidate(self, pos_class, num_class, num_ins_per_class):
        '''
        random pick some instances for snowball phase 2 with total number num_class (1 pos + num_class-1 neg) * num_ins_per_class
        pos_class: positive relation (name)
        num_class: total number of classes, including the positive and negative relations
        num_ins_per_class: the number of instances of each relation
        return: a dataset
        '''

        target_classes = random.sample(self.dataset.rel2scope.keys(), num_class)
        if not pos_class in target_classes:
            target_classes = target_classes[:-1] + [pos_class]
        candidate = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': [], 'entpair': []}

        for i, class_name in enumerate(target_classes):
            scope = self.dataset.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), min(num_ins_per_class, scope[1] - scope[0]),
                                       True)
            candidate['word'].append(self.dataset.data_word[indices])
            candidate['posh'].append(self.dataset.data_posh[indices])
            candidate['post'].append(self.dataset.data_post[indices])
            candidate['mask'].append(self.dataset.data_mask[indices])
            candidate['id'].append(self.dataset.uid[indices])
            candidate['entpair'] += list(self.dataset.data_entpair[indices])

        for key in ['word', 'posh', 'post', 'mask', 'id']:
            candidate[key] = np.concatenate(candidate[key], 0)
            candidate[key] = Variable(torch.from_numpy(candidate[key]).long())
            if args.cuda:  candidate[key] = candidate[key].cuda() # To cuda

        return candidate

    def sample_for_eval(self, train_dataloader, support_pos_size, query_size,total_iter,
                        target_class=None, query_train=True,query_val=True):
        #res=[]
        for step in range(total_iter):
            if target_class is None:
                total_class=np.random.choice(list(self.dataset.rel2scope.keys()),2,replace=False)
                target_class,neg_class=total_class[0],total_class[1]
                #target_class = random.sample(self.dataset.rel2scope.keys(), 1)[0]
            else:
                neg_class=None
                while neg_class!=target_class:
                    neg_class = random.sample(self.dataset.rel2scope.keys(), 1)[0]

            support_pos = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': [], 'entpair': []}
            support_neg = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': [], 'entpair': []}
            query = {'word': [], 'posh': [], 'post': [], 'mask': [], 'id': [], 'label': []}

            # Negative ralation
            scope_neg=self.dataset.rel2scope[neg_class]
            indices_neg= np.random.choice(list(range(scope_neg[0], scope_neg[1])),support_pos_size, True)
            support_word_neg=self.dataset.data_word[indices_neg]
            support_posh_neg,support_post_neg = self.dataset.data_posh[indices_neg],self.dataset.data_post[indices_neg]
            support_mask_neg,support_id_neg=self.dataset.data_mask[indices_neg],self.dataset.uid[indices_neg]
            support_entpair_neg=list(self.dataset.data_entpair[indices_neg])

            support_neg['word'] = support_word_neg
            support_neg['posh'], support_neg['post'] = support_posh_neg, support_post_neg
            support_neg['mask'], support_neg['label'] = support_mask_neg, np.zeros((support_pos_size), dtype=np.int32)
            support_neg['id'], support_neg['entpair'] = support_id_neg, support_entpair_neg

            # New relation
            scope = self.dataset.rel2scope[target_class]
            indices = np.random.choice(list(range(scope[0], scope[1])), support_pos_size + query_size, True)
            support_word, query_word, _ = np.split(self.dataset.data_word[indices],
                                                   [support_pos_size, support_pos_size + query_size])
            support_posh, query_posh, _ = np.split(self.dataset.data_posh[indices],
                                                   [support_pos_size, support_pos_size + query_size])
            support_post, query_post, _ = np.split(self.dataset.data_post[indices],
                                                   [support_pos_size, support_pos_size + query_size])
            support_mask, query_mask, _ = np.split(self.dataset.data_mask[indices],
                                                   [support_pos_size, support_pos_size + query_size])
            support_id, query_id,_=np.split(self.dataset.uid[indices],[support_pos_size, support_pos_size + query_size])
            support_entpair = list(self.dataset.data_entpair[indices[:support_pos_size]])

            support_pos['word'] = support_word
            support_pos['posh'],support_pos['post'] = support_posh,support_post
            support_pos['mask'],support_pos['label'] = support_mask,np.ones((support_pos_size), dtype=np.int32)
            support_pos['id'],support_pos['entpair'] = support_id,support_entpair

            query['word'].append(query_word)
            query['posh'].append(query_posh)
            query['post'].append(query_post)
            query['mask'].append(query_mask)
            query['id'].append(query_id)
            query['label'] += [1] * query_size

            # from train data loader
            if query_train:
                for i, class_name in enumerate(train_dataloader.dataset.rel2scope.keys()):
                    scope = train_dataloader.dataset.rel2scope[class_name]
                    indices = np.random.choice(list(range(scope[0], scope[1])), query_size, True)
                    query['word'].append(train_dataloader.dataset.data_word[indices])
                    query['posh'].append(train_dataloader.dataset.data_posh[indices])
                    query['post'].append(train_dataloader.dataset.data_post[indices])
                    query['mask'].append(train_dataloader.dataset.data_mask[indices])
                    query['id'].append(train_dataloader.dataset.uid[indices])
                    query['label'] += [0] * query_size
            # from current data loader
            if query_val:
                for i, class_name in enumerate(self.dataset.rel2scope.keys()):
                    if class_name == target_class: continue
                    scope = self.dataset.rel2scope[class_name]
                    indices = np.random.choice(list(range(scope[0], scope[1])), query_size, True)
                    query['word'].append(self.dataset.data_word[indices])
                    query['posh'].append(self.dataset.data_posh[indices])
                    query['post'].append(self.dataset.data_post[indices])
                    query['mask'].append(self.dataset.data_mask[indices])
                    query['id'].append(self.dataset.uid[indices])
                    query['label'] += [0] * query_size

            for key in ['word', 'posh', 'post', 'mask', 'id']: query[key] = np.concatenate(query[key], 0)
            query['label'] = np.array(query['label'])

            for key in ['word', 'posh', 'post', 'mask', 'label', 'id']:
                query[key]=Variable(torch.from_numpy(query[key]).long())
                support_pos[key] = Variable(torch.from_numpy(support_pos[key]).long())
                support_neg[key] = Variable(torch.from_numpy(support_neg[key]).long())
                if args.cuda: # To cuda
                    query[key] = query[key].cuda()
                    support_pos[key] = support_pos[key].cuda()
                    support_neg[key] = support_neg[key].cuda()

            #res.append([step,support_pos,support_neg, query, target_class,neg_class])
            yield step,support_pos,support_neg, query, target_class,neg_class
        #return res



if __name__=="__main__":
    data=REDataset(args.train_data_file_path,
                   args.word_vector_file_path,
                   args.mode,
                   args.case_sensitive,
                   args.max_length,
                   args.cuda,
                   args.reprocess)
    dataloader=REDataLoader(dataset=data,batch_size=200,shuffle=args.shuffle,sampler=None)
    for epoch in range(5):
        for batch in dataloader:
            x,y=batch,batch["label"]
            print(x,y)
