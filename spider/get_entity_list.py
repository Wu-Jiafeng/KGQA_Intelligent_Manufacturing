import codecs
def get_entities():
    f = codecs.open('../triple_data/output.txt','r','utf-8')
    data = []
    for line in f.readlines():
        array = line.strip("\n").split("\t")
        arr = [array[0],array[2]]
        data.extend(arr)
    
    return data

