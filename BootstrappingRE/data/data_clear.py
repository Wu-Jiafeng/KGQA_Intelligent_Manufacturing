import os
from tqdm import tqdm

def strQ2B(ustring):
    # """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)  ##注意这里要是前面加一个tab，就不会将if里的内容加进来了
    return rstring  ##注意这里要是前面加一个tab，就变成第一次就return了

def duie_data_vlear(): # error
    train, val, test = \
        r"D:/学校/毕业论文/data/duie_data/duie_ori_train.json", \
        r"D:/学校/毕业论文/data/duie_data/duie_ori_dev.json", \
        r"D:/学校/毕业论文/data/duie_data/duie_ori_test.json"
    for filename in [train, val, test]:
        f = open(filename, "r", encoding="utf-8")
        fw = open(filename + ".json", "a", encoding="utf-8")
        text = f.readlines()
        for t in tqdm(text):
            t = strQ2B(t).lower()
            fw.write(t)

def txt_data_clear():
    for root, dirs, files in os.walk(r"D:/学校/毕业论文/data/TXT"):
        for file in files:
            f = open(os.path.join(root, file), "r", encoding="utf-8")
            text = f.read()
            text = strQ2B(text).replace(" ","").replace("。","。\n").lower()
            fw = open(os.path.join("D:/学校/毕业论文/data/output", file), "w", encoding="utf-8")
            fw.write(text)

if __name__=="__main__":
    #duie_data_vlear()
    txt_data_clear()