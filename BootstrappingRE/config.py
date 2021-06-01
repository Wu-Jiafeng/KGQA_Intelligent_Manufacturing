from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--m",
                    help="train_encoder/train_sim/test",default="test")
parser.add_argument("--me",
                    help="cnn/pcnn",default="cnn")
parser.add_argument("--ms",
                    help="sim/tri",default="sim")
parser.add_argument("--processed_data_dir",
                    help="the directory that stores processed data",default="./data/processed_data")
# data file path
parser.add_argument("--train_data_file_path",
                    help="the path to the dataset file",default="./data/duie_train.json")
parser.add_argument("--val_data_file_path",
                    help="the path to the dataset file",default="./data/duie_val.json")
parser.add_argument("--test_data_file_path",
                    help="the path to the dataset file",default="./data/duie_test.json")
parser.add_argument("--seed_data_file_path",
                    help="the path to the seed file",default="./data/seed.json")
parser.add_argument("--unlabeled_data_file_path",
                    help="the path to the unlabeled file",default="./data/unlabeled.json")
parser.add_argument("--seed1_data_file_path",
                    help="the path to the seed file",default="./data/seed1.json")
parser.add_argument("--unlabeled1_data_file_path",
                    help="the path to the unlabeled file",default="./data/unlabeled1.json")
parser.add_argument("--word_vector_file_path",
                    help="the path to the word vector file",default="./data/word2vec.baidubaike.dim300.bin2")
parser.add_argument("--ckpt_dir",
                    help="directory that saves checkpoints",default="./checkpoints")
# dataset related config
parser.add_argument("--mode",
                    help="Train/Val/Test",default="Train")
parser.add_argument("--test_all",
                    help="check if test all model",type=bool,default=False)
parser.add_argument("--case_sensitive",
                    help="check if the dataset is case-sensitive",type=bool,default=True)
parser.add_argument("--max_length",
                    help="the max length of a instance/sentence",type=int,default=40)
parser.add_argument("--reprocess", # 改了max_length就要reprocess
                    help="decide if the dataset needs reprocessing",type=bool,default=False)
parser.add_argument("--word_embedding_dim",
                    help="the dimension of word embedding",type=int,default=300)
parser.add_argument("--pos_embedding_dim",
                    help="the dimension of position embedding",type=int,default=5)
parser.add_argument("--cuda",
                    help="decide if the model needs training with GPU",type=bool,default=True)
parser.add_argument("--shuffle",
                    help="decide if the dataset needs shuffling",type=bool,default=True)
# model related config
parser.add_argument("--warmup_step",
                    help="warm up step when training encoder",type=int,default=300)
parser.add_argument("--grad_iter",
                    help="grad_iter",type=int,default=1)
parser.add_argument("--hidden_size",
                    help="data hidden size",type=int,default=230)
parser.add_argument("--encoder_type",
                    help="sentence encoder type name (CNN/PCNN)",default="CNN")
parser.add_argument("--encoder_name",
                    help="sentence encoder saved name",default="cnn_encoder_on_fewrel")
parser.add_argument("--encoder_batch_size",
                    help="batch size when training encoder",type=int,default=32)
parser.add_argument("--encoder_training_epoch",
                    help="epoch when training encoder",type=int,default=100)
parser.add_argument("--encoder_training_lr",
                    help="learning rate when training encoder",type=float,default=1.)
parser.add_argument("--encoder_pretrained",
                    help="the encoder pretrained file",default=None)
parser.add_argument("--similarity_network_type",
                    help="similarity network type name (Siamese/Triplet)",default="Siamese")
parser.add_argument("--similarity_network_name",
                    help="similarity network saved name",default="cnn_siamese_on_fewrel")
parser.add_argument("--similarity_network_batch_size",
                    help="batch size when training similarity network",type=int,default=128)
parser.add_argument("--similarity_network_multiclass_size",
                    help="multi-class size when training similarity network",type=int,default=8)
parser.add_argument("--similarity_network_training_iter",
                    help="iteration when training similarity network",type=int,default=30000)
parser.add_argument("--similarity_network_training_val_iter",
                    help="validation iteration when training similarity network",type=int,default=2000)
parser.add_argument("--similarity_network_training_lr",
                    help="learning rate when training similarity network",type=float,default=1.)
parser.add_argument("--similarity_network_pretrained",
                    help="the similarity network pretrained file",default=None)
# snowball hyperparameter
parser.add_argument('--shot',
                    default=5, type=int,help='Number of seeds')
parser.add_argument('--snowball_eval_iter',
                    default=1000, type=int,help='Eval iteration')
parser.add_argument("--phase1_add_num",
                    help="number of instances added in phase 1", type=int, default=5)
parser.add_argument("--phase2_add_num",
                    help="number of instances added in phase 2", type=int, default=5)
parser.add_argument("--phase1_siamese_th",
                    help="threshold of relation siamese network in phase 1", type=float, default=0.9)
parser.add_argument("--phase2_siamese_th",
                    help="threshold of relation siamese network in phase 2", type=float, default=0.9)
parser.add_argument("--phase2_cl_th",
                    help="threshold of relation classifier in phase 2", type=float, default=0.9)
parser.add_argument("--snowball_max_iter",
                    help="number of iterations of snowball", type=int, default=5)
# inference batch_size
parser.add_argument("--infer_batch_size",
                    help="batch size when inference", type=int, default=0)
# fine-tune hyperparameter
parser.add_argument("--finetune_epoch",
                    help="num of epochs when finetune", type=int, default=50)
parser.add_argument("--finetune_batch_size",
                    help="batch size when finetune", type=int, default=10)
parser.add_argument("--finetune_lr",
                    help="learning rate when finetune", type=float, default=0.05)
parser.add_argument("--finetune_wd",
                    help="weight decay rate when finetune", type=float, default=1e-5)
parser.add_argument("--finetune_weight",
                    help="loss weight of negative samples", type=float, default=0.2)
# print
parser.add_argument("--print_debug",
                    help="print debug information", type=bool,default=True)
parser.add_argument("--eval",
                    help="eval during snowball", type=bool,default=True)

