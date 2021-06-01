import os
import struct
import ujson as json

# no need to run again!
def word_vector2json():
    wv_file,wv_bin_file=r"./sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5",\
                        r"./word2vec.baidubaike.dim300.bin2"
    wv_data,wv_bin=open(wv_file,"r",encoding="utf-8"),open(wv_bin_file,"wb")
    # get word num and vector dim
    line=wv_data.readline()
    w_num,wv_dim=line.strip().split(" ")
    wv_bin.write(struct.pack('ii',int(w_num),int(wv_dim)))
    print(w_num,wv_dim)
    # get word_vector dict
    line=wv_data.readline()
    while line:
        wordNvec=line.strip().split(" ")
        word,vec=wordNvec[0],wordNvec[1:]
        word = word.encode("utf-8")
        word_len = len(word)
        wv_bin.write(struct.pack('i', word_len))
        wv_bin.write(word)
        for v in vec:
            wv_bin.write(struct.pack("f", float(v)))
        line=wv_data.readline()
    print("The word_vector dict has been built and stored!")

if __name__=="__main__":
    word_vector2json()