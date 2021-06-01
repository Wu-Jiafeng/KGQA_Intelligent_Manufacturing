import os
import jieba
import ujson as json
import random
from tqdm import tqdm
from BootstrappingRE.data.data_clear import strQ2B

# for original duie data, now no need to run this program
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

def data_merge():
    '''
            file_name: Json file storing the data in the following format
                {
                    "P155": # relation id
                        [
                            {
                                "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                                "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                                "token": ["Hot", "Dance", "Club", ...], # sentence
                            },
                            ...
                        ],
                    "P177":
                        [
                            ...
                        ]
                    ...
                }
    '''
    # load the original data files
    print("Now load the files!")
    rel_file=r"./duie_data/duie_schema.json"
    train_file,dev_file,test_file=\
        r"./duie_data/duie_ori_train.json",r"./duie_data/duie_ori_dev.json","./duie_data/duie_ori_test.json"
    rel_data=json.load(open(rel_file,'r',encoding="utf-8"))
    train_data,dev_data,test_data=\
        json.load(open(train_file,'r',encoding="utf-8")),\
        json.load(open(dev_file,'r',encoding="utf-8")),\
        json.load(open(test_file,'r',encoding="utf-8"))
    print("Finish Loading files! Now merge the data!")

    # construct relation dict and vocab (如果分词效果不好改用字向量)
    rel_dict,vocab={},set()
    for instance in rel_data: rel_dict[instance["predicate"]]=[]
    for instance in tqdm(train_data+dev_data+test_data):
        # sentence tokenization and head/tail entity processing
        if "spo_list" not in instance.keys(): continue
        text=strQ2B(instance["text"])
        tokens_pos=list(jieba.tokenize(text))
        tokens=[token for (token,_,_) in tokens_pos]
        #tokens=list(text)
        #for token in tokens: vocab.add(token)
        spo_list=instance["spo_list"]
        for triple in spo_list:
            s,p,o=[strQ2B(triple["subject"]),triple["subject_type"]],\
                  triple["predicate"],\
                  [strQ2B(triple["object"]["@value"]),triple["object_type"]["@value"]]
            # get entity position information, then match them to tokens
            if s[0]=='' or o[0]=='': continue
            s_index_blocks,o_index_blocks=get_index_blocks(s[0],text),get_index_blocks(o[0],text)
            #s.append(s_index_blocks)
            #o.append(o_index_blocks)
            s.append([[] for _ in s_index_blocks])
            o.append([[] for _ in o_index_blocks])
            s_block,o_block,len_s_block,len_o_block=0,0,len(s_index_blocks),len(o_index_blocks)
            for i in range(len(tokens_pos)):
                (token,posb,pose)=tokens_pos[i]
                pose_fit=pose-1
                vocab.add(token)
                # head entity position match
                if s_block < len_s_block:
                    if (posb in s_index_blocks[s_block] or pose_fit in s_index_blocks[s_block]):
                        s[2][s_block].append(i)
                        if posb <= s_index_blocks[s_block][-1] <= pose_fit: s_block += 1
                    elif posb < s_index_blocks[s_block][0] and s_index_blocks[s_block][-1] < pose_fit:
                        s[2][s_block].append(i)
                        s_block += 1

                # tail entity position match
                if o_block < len_o_block:
                    if (posb in o_index_blocks[s_block] or pose_fit in o_index_blocks[s_block]):
                        o[2][o_block].append(i)
                        if posb <= o_index_blocks[o_block][-1] <= pose_fit: o_block += 1
                    elif posb < o_index_blocks[o_block][0] and o_index_blocks[o_block][-1] < pose_fit:
                        o[2][o_block].append(i)
                        o_block += 1

            if len(s[2][0])==0 or len(o[2][0])==0:
                print("Error")

            # append to the rel_dict
            rel_dict[p].append({"h": s,"t": o,"token":tokens})
    print("The data has been merged! Now store it!")

    all_file,vocab_file=r"./duie_all.json",r"./duie_vocab.json"
    json.dump(rel_dict,open(all_file,'w',encoding="utf-8"),ensure_ascii=False)
    json.dump(dict(zip(list(vocab),list(range(len(vocab))))),
              open(vocab_file,'w',encoding="utf-8"),ensure_ascii=False)

    print("Finish storing!")

def data_split():
    # load the original data files
    print("Now load the data!")
    all_file=r"./duie_all.json"
    all_data=json.load(open(all_file,'r',encoding="utf-8"))
    train_data,val_data,test_data={},{},{}
    print("Finish loading the data! Now split the data!")

    #split the data
    for key in all_data.keys(): train_data[key],val_data[key],test_data[key]=[],[],[]
    for key,instances in all_data.items():
        for instance in instances:
            p=random.random()
            # train:val:test = 8:1:1
            if p<=0.8: train_data[key].append(instance)
            elif p<=0.9: val_data[key].append(instance)
            else: test_data[key].append(instance)
    for key in list(val_data.keys()):
        if len(val_data[key])==0: del val_data[key]
    for key in list(test_data.keys()):
        if len(test_data[key]) == 0: del test_data[key]
    print("Finish spliting the data! Now store the data expectively!")

    # store the data
    train_file,val_file,test_file=\
        r"./duie_train.json",r"./duie_val.json",r"./duie_test.json"
    json.dump(train_data, open(train_file, 'w', encoding="utf-8"), ensure_ascii=False)
    json.dump(val_data, open(val_file, 'w', encoding="utf-8"), ensure_ascii=False)
    json.dump(test_data, open(test_file, 'w', encoding="utf-8"), ensure_ascii=False)
    print("Finish storing!")

def data_relation_clear():
    val_file, test_file =  r"./duie_val.json", r"./duie_test.json"
    val_data,test_data=json.load(open(val_file,'r',encoding="utf-8")),json.load(open(test_file,'r',encoding="utf-8"))
    for key in list(val_data.keys()):
        if len(val_data[key])==0: del val_data[key]
    for key in list(test_data.keys()):
        if len(test_data[key]) == 0: del test_data[key]

    val_file_w, test_file_w = r"./duie_val0.json", r"./duie_test0.json"
    json.dump(val_data, open(val_file_w, 'w', encoding="utf-8"), ensure_ascii=False)
    json.dump(test_data, open(test_file_w, 'w', encoding="utf-8"), ensure_ascii=False)
    print("Finish storing!")

if __name__=="__main__":
    data_merge()
    data_split()
    #data_relation_clear()