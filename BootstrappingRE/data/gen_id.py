import numpy as np
import json
import os
from BootstrappingRE.config import parser

args=parser.parse_args()
processed_data_dir=args.processed_data_dir
train = json.load(open('./duie_train.json', "r",encoding="utf-8"))
val = json.load(open('./duie_val.json', "r",encoding="utf-8"))
test = json.load(open('./duie_test.json', "r",encoding="utf-8"))
seed = json.load(open('./seed.json', "r",encoding="utf-8"))
unlabeled=json.load(open('./unlabeled.json', "r",encoding="utf-8"))
#seed1 = json.load(open(args.seed1_data_file_path, "r",encoding="utf-8"))
#unlabeled1=json.load(open(args.unlabeled1_data_file_path, "r",encoding="utf-8"))
#distant = json.load(open('./distant.json'))

def gen_id():
    total = 0
    for data, name in [(train, 'duie_train'), (val, 'duie_val'), (test, 'duie_test'),(seed, 'seed'),(unlabeled,'unlabeled')]:#,(seed1, 'seed1'),(unlabeled1,'unlabeled1')]:
        print(name)
        count = 0
        for rel in data:
            count += len(data[rel])
        data_id = np.array(list(range(total, total + count)), dtype=np.int32)
        np.save(os.path.join(processed_data_dir, name + '_uid.npy'), data_id)
        total += count

if __name__=="__main__":
    gen_id()

