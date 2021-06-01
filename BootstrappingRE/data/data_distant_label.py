import os
import re
# no need to run again!
ESYNTAX = r'\[\@(.*?)\#.*?\*\](?!\#)'# 不带括号
ERSYNTAX = r'\[\@.*?\#(.*?)\*\](?!\#)'# 实体类别
EBSYNTAX=r'\[\@.*?\#.*?\*\](?!\#)' #带括号
EFDIR=r"./im_data/ori_data"
TFDIR=r"./im_data/TXT"
WFDIR=r"./im_data/labeled_data"

def getCurrentEntitySet(fdir):
    EntityDict={}
    for root,dirs,files in os.walk(fdir):
        for file in files:
            f = open(os.path.join(root, file), "r", encoding="utf-8")
            text = f.read().lower().split("\n")
            for t in text:
                t=t.strip().split("\t\t")[-1]
                if t=='': continue
                ebs=re.findall(EBSYNTAX,t)
                for eb in ebs:
                    #e,er=re.findall(ESYNTAX,eb)[0],re.findall(ERSYNTAX,eb)[0]
                    EntityDict[re.findall(ESYNTAX,eb)[0]]='\t'.join(list(eb))
    return sorted(EntityDict.items(),key=lambda kv: len(kv[0]),reverse=True)

def EntityTransform(tdir,wdir,EntityDict):
    for root, dirs, files in os.walk(tdir):
        for file in files:
            f = open(os.path.join(root, file), "r", encoding="utf-8")
            text = f.read().lower()
            for key,value in EntityDict: text=text.replace(key,value)
            text=text.replace('\t','')
            fw = open(os.path.join(wdir, file+'.ann'), "w", encoding="utf-8")
            fw.write(text)

if __name__=="__main__":
    EntityDict=getCurrentEntitySet(EFDIR)
    #EntityTransform(TFDIR,WFDIR,EntityDict)