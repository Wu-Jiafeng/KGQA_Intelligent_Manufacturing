import os
import jieba
import ujson as json
import numpy as np
import random
import re
from tqdm import tqdm
from BootstrappingRE.data.data_clear import strQ2B

#no need to run again!
def get_index_blocks(pattern,text):
    index_blocks=[]
    sentence=text
    pos,posb,lenp=0,text.find(pattern),len(pattern)
    while posb!=-1:
        pos+=posb
        index_blocks.append(list(range(pos,pos+lenp)))
        sentence=sentence[posb+lenp:]
        pos+=lenp
        posb=sentence.find(pattern)
    return index_blocks

def get_entity_dict(s):
    sentence,entity2type=s,{}
    ESYNTAX = r'\[\@(.*?)\#.*?\*\](?!\#)'  # 不带括号
    RESYNTAX = r'\[\$(.*?)\#.*?\*\](?!\#)'  # 不带括号
    ERSYNTAX = r'\[\@.*?\#(.*?)\*\](?!\#)'  # 实体类别
    RERSYNTAX = r'\[\$.*?\#(.*?)\*\](?!\#)'  # 实体类别
    EBSYNTAX = r'\[\@.*?\#.*?\*\](?!\#)'  # 带括号
    REBSYNTAX = r'\[\$.*?\#.*?\*\](?!\#)'
    ebs = re.findall(EBSYNTAX, s)
    for eb in ebs:
        word,type=re.findall(ESYNTAX,eb)[0],re.findall(ERSYNTAX,eb)[0]
        entity2type[word]=type
        sentence=sentence.replace(eb,word)
    rebs=re.findall(REBSYNTAX, s)
    for reb in rebs:
        word,type=re.findall(RESYNTAX,reb)[0],re.findall(RERSYNTAX,reb)[0]
        entity2type[word] = type
        sentence=sentence.replace(reb,word)
    return entity2type,sentence

def get_seed_set():
    # seed set encoded as duie_type
    rel_dict={"应用于":[],"关注执行":[],"学习借鉴":[],
          "区别于":[],"基础来源":[],"事件地点":[],
          "事件时间":[],"技术特点":[],"问题挑战":[]}
    unlabel_dict = {"应用于":[],"关注执行":[],"学习借鉴":[],
          "区别于":[],"基础来源":[],"事件地点":[],
          "事件时间":[],"技术特点":[],"问题挑战":[]}
    for root, dirs, files in os.walk(r"./im_data/ori_data"):
        for file in files:
            lines = open(os.path.join(root, file), "r", encoding="utf-8-sig").readlines()
            for line in lines:
                line = line.replace("\ufeff","").strip().split("\t\t")
                if len(line[-1]) == 0: continue
                if len(line)>1:
                    triples,sentence=line[:-1],line[-1]
                    entity2type,sentence=get_entity_dict(sentence)
                    # sentence tokenization
                    sentence=strQ2B(sentence)
                    tokens_pos = list(jieba.tokenize(sentence))
                    tokens = [token for (token, _, _) in tokens_pos]
                    for triple in triples:
                        s,p,o=triple.split("\t")
                        s,o=[s,entity2type[s]],[o,entity2type[o]]
                        if s[0] == '' or o[0] == '': continue
                        s_index_blocks, o_index_blocks = get_index_blocks(s[0], sentence), get_index_blocks(o[0], sentence)
                        # s.append(s_index_blocks)
                        # o.append(o_index_blocks)
                        s_index_list, o_index_list = [i for block in s_index_blocks for i in block], \
                                                     [i for block in o_index_blocks for i in block]
                        s.append([[] for _ in s_index_blocks])
                        o.append([[] for _ in o_index_blocks])
                        s_block, o_block, len_s_block, len_o_block = 0, 0, len(s_index_blocks), len(o_index_blocks)
                        for i in range(len(tokens_pos)):
                            (token, posb, pose) = tokens_pos[i]
                            pose_fit = pose - 1
                            # head entity position match
                            if s_block < len_s_block and (posb in s_index_list or pose_fit in s_index_list):
                                s[2][s_block].append(i)
                                if posb <= s_index_blocks[s_block][-1] <= pose_fit: s_block += 1
                            elif s[0] in token:
                                s[2][s_block].append(i)
                                s_block += 1

                            # tail entity position match
                            if o_block < len_o_block and (posb in o_index_list or pose_fit in o_index_list):
                                o[2][o_block].append(i)
                                if posb <= o_index_blocks[o_block][-1] <= pose_fit: o_block += 1
                            elif o[0] in token:
                                o[2][o_block].append(i)
                                o_block += 1

                        if len(s[2][0]) == 0 or len(o[2][0]) == 0:
                            print("Error")

                        # append to the rel_dict
                        rel_dict[p].append({"h": s, "t": o, "token": tokens})
                else:
                    entityinfo=[]
                    entity2type, sentence = get_entity_dict(line[0])
                    tokens_pos = list(jieba.tokenize(sentence))
                    tokens = [token for (token, _, _) in tokens_pos]
                    for e,t in entity2type.items():
                        info=[e, t]
                        e_index_blocks=get_index_blocks(e, sentence)
                        info.append([[] for _ in e_index_blocks])
                        e_block,len_e_block= 0,len(e_index_blocks)
                        for i in range(len(tokens_pos)):
                            (token, posb, pose) = tokens_pos[i]
                            pose_fit = pose - 1
                            # head entity position match
                            if e_block < len_e_block:
                                if (posb in e_index_blocks[e_block] or pose_fit in e_index_blocks[e_block]):
                                    info[2][e_block].append(i)
                                    if posb <= e_index_blocks[e_block][-1] <= pose_fit: e_block += 1
                                elif posb<e_index_blocks[e_block][0]and e_index_blocks[e_block][-1]<pose_fit:
                                    info[2][e_block].append(i)
                                    e_block += 1
                        if len(info[2][0]) == 0: print("Error!")
                        if info[2][0][0]<40: entityinfo.append(info)
                    # create candidate instance
                    for info1 in entityinfo:
                        for info2 in entityinfo:
                            if info1==info2:continue
                            label=random.choice(list(unlabel_dict.keys()))
                            unlabel_dict[label].append({"h":info1,"t":info2,"token":tokens})
                    #unlabel_list.append({"entities":entityinfo,"token":tokens})

    seed_file= r"./seed.json"
    unlabeled_file=r"./unlabeled.json"
    #unlabeled_file=open(r"D:/学校/毕业论文/BootstrappingRE/data/im_data/unlabeled.json",'a',encoding="utf-8")
    json.dump(rel_dict, open(seed_file, 'w', encoding="utf-8"), ensure_ascii=False)
    json.dump(unlabel_dict, open(unlabeled_file, 'w', encoding="utf-8"), ensure_ascii=False)
    #unlabeled_file.write("[")
    #for instance in unlabel_list[:-1]: unlabeled_file.write(str(instance)+',\n')
    #unlabeled_file.write(str(unlabel_list[-1])+']')
    print("Finish storing!")

def seed_data_preprocess(filename):
    # loading the original data...
    print("Loading the original data file...")
    processed_data_dir="./processed_data"
    ori_data = json.load(open("./"+filename+".json", "r", encoding="utf-8"))
    word2id = json.load(open("./processed_data/word2vec.baidubaike.dim300_word2id.json", "r", encoding="utf-8"))
    print("Finish loading")

    # Pre-process data
    print("Pre-processing data...")
    instance_tot = 0
    for relation in ori_data:
        instance_tot += len(ori_data[relation])
    data_word = np.zeros((instance_tot, 40), dtype=np.int32)
    data_posh = np.zeros((instance_tot, 40), dtype=np.int32)
    data_post = np.zeros((instance_tot, 40), dtype=np.int32)
    data_mask = np.zeros((instance_tot, 40), dtype=np.int32)
    data_length = np.zeros((instance_tot), dtype=np.int32)
    data_label = np.zeros((instance_tot), dtype=np.int32)
    data_entpair = []
    rel2scope = {}  # left close right open
    entpair2scope = {}
    rel2id = {}
    rel_tot = len(ori_data)
    i = 0
    for cur_id, (relation, instance_list) in tqdm(enumerate(ori_data.items())):
        rel2scope[relation] = [i, i]
        rel2id[relation] = cur_id
        for ins in instance_list:
            tokens = ins['token']
            head, tail = ins['h'][0], ins['t'][0]
            posh, post = ins['h'][2][0][0], ins['t'][2][0][0]
            data_word2id = data_word[i]
            entpair = head + '#' + tail
            data_entpair.append(entpair)  # record entity pair per instance

            # map the tokens to word2id
            for j, token in enumerate(tokens):
                if j < 40:
                    if token in word2id:
                        data_word2id[j] = word2id[token]
                    else:
                        data_word2id[j] = word2id['UNK']
            for j in range(len(tokens), 40): data_word2id[j] = word2id['BLANK']

            # get the relative position (to head or tail) of tokens and divide them into 4 parts:
            # ---part1----entity1----part2----entity2-----part3-----maxlength-----part0-----
            data_label[i] = rel2id[relation]
            data_length[i] = min(len(tokens), 40)
            posh, post = min(posh, 40 - 1), min(post, 40 - 1)
            pos_min, pos_max = min(posh, post), max(posh, post)
            for j in range(40):
                data_posh[i][j] = j - posh + 40
                data_post[i][j] = j - post + 40
                if j >= data_length[i]:
                    data_mask[i][j] = 0
                elif j <= pos_min:
                    data_mask[i][j] = 1
                elif j <= pos_max:
                    data_mask[i][j] = 2
                else:
                    data_mask[i][j] = 3

            # get entity pair instance set scope
            if not entpair in entpair2scope:
                entpair2scope[entpair] = [i]
            else:
                entpair2scope[entpair].append(i)
            i += 1
        # get relation instance set scope
        rel2scope[relation][1] = i

    word_npy_file_name = os.path.join(processed_data_dir, filename + '_word.npy')
    posh_npy_file_name = os.path.join(processed_data_dir, filename + '_posh.npy')
    post_npy_file_name = os.path.join(processed_data_dir, filename + '_post.npy')
    mask_npy_file_name = os.path.join(processed_data_dir, filename + '_mask.npy')
    length_npy_file_name = os.path.join(processed_data_dir, filename + '_length.npy')
    label_npy_file_name = os.path.join(processed_data_dir, filename + '_label.npy')
    entpair_npy_file_name = os.path.join(processed_data_dir, filename + '_entpair.npy')
    rel2scope_file_name = os.path.join(processed_data_dir, filename + '_rel2scope.json')
    rel2id_file_name = os.path.join(processed_data_dir, filename + '_rel2id.json')
    entpair2scope_file_name = os.path.join(processed_data_dir, filename + '_entpair2scope.json')
    # store files
    print("Finish pre-processing! Now Store files!")
    data_entpair = np.array(data_entpair)
    np.save(word_npy_file_name, data_word)
    np.save(posh_npy_file_name, data_posh)
    np.save(post_npy_file_name, data_post)
    np.save(mask_npy_file_name, data_mask)
    np.save(length_npy_file_name, data_length)
    np.save(label_npy_file_name, data_label)
    np.save(entpair_npy_file_name, data_entpair)
    json.dump(rel2scope, open(rel2scope_file_name, "w", encoding="utf8"), ensure_ascii=False)
    json.dump(rel2id, open(rel2id_file_name, "w", encoding="utf8"), ensure_ascii=False)
    json.dump(entpair2scope, open(entpair2scope_file_name, "w", encoding="utf8"), ensure_ascii=False)
    print("Finish storing")

def bulide2tdict():
    seed,unlabeled=json.load(open("./seed.json", "r", encoding="utf-8")),json.load(open("./unlabeled.json", "r", encoding="utf-8"))
    e2t={}
    for key in seed.keys():
        instances_s,instances_u=seed[key],unlabeled[key]
        for instance in instances_s+instances_u:
            h,t=instance["h"],instance["t"]
            e2t[h[0]]=h[1]
            e2t[t[0]] = t[1]
    json.dump(e2t,open('../../triple_data/e2t.json', 'w', encoding="utf-8"), ensure_ascii=False)

if __name__=="__main__":
    #data_relation_clear()
    get_seed_set()
    #seed_data_preprocess("seed")
    #seed_data_preprocess("unlabeled")
    #bulide2tdict()
    