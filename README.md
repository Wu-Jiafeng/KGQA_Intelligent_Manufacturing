# KGQA_Intelligent Manufacturing：面向智能制造的知识图谱抽取方法研究及其应用

## 目录树:<br>
1)  app.py：知识图谱可视化及问答系统的主入口<br>
2)  BootstrappingRE：基于度量学习的自举关系抽取模块<br>
	 |-main.py：关系抽取模块主入口，包含编码器训练、相似性网络训练、整体测试及预测<br>
	 |-config.py：模型及数据集相关参数<br>
	 |-checkpoints文件夹：存储训练模型参数文件<br>
	 |-data文件夹：存储相关数据(相关数据已被压缩为data.zip)<br>
	　　 |-data_clear.py：对duie数据集及智能制造语料进行数据清洗<br>
	　　 |-data_distant_label.py：对智能制造语料进行远程监督实体标注<br>
	　　 |-dataset.py：定义模型输入数据集细节<br>
	　　 |-*_type_data_process.py：将duie数据集和智能制造数据集进行划分并整理为FewRel数据集格式<br>
	　　 |-word_vector.py：将原始词向量文件转成二进制文件，释放空间<br>
	　　 |-duie_*.json：duie数据集文件<br>
	　　 |-seed.json：智能制造语料种子集<br>
	　　 |-unlabeled.json：智能制造语料未标注数据<br>
	　　 |-duie_data文件夹：duie数据集原始数据<br>
	　　 |-im_data文件夹：智能制造语料原始数据<br>
	　　　　 |-labeled_data文件夹：经远程实体标注后的数据<br>
	　　　　 |-ori_data文件夹：未经远程实体标注后的数据<br>
	　　　　 |-TXT文件夹：195篇经知网智能制造相关的论文PDF转换并清洗的TXT文件<br>
	　　 |-processed_data文件夹：数据集及词向量经预处理的数据文件<br>
	 |-models文件夹：模型相关算法<br>
	　　 |-bootstrapping.py：基于度量学习的自举关系抽取模块及基准算法模块<br>
	　　 |-encoder.py：编码器模块，已实现CNN及PCNN编码器<br>
	　　 |-similarity_network.py：相似性度量网络模块，已实现Simaese及Triplet网络<br>
	 |-output文件夹：模型输出，如三元组文件、相似性网络训练效果文件<br>
2)  KGQA文件夹：问答系统句法分析模块<br>
	 |-model文件夹：存放pyLTP包中的cws及pos语言模型<br>
     |-bilstm_ltp.py 基于bilstm-crf和ltp混合的分词、词性标注、命名实体识别<br>
3)  neo_db文件夹：知识图谱构建模块<br>
     |-config.py 相关配置，包括neo4j登录、问答相似规则表、生成BiLSTM-CRF NER模型实例<br>
     |-create_graph.py 将triple_data中的数据导入Neo4j图数据库<br>
     |-query_graph.py 实现图数据库相关查询操作及生成数据库全实体关系json文件<br>
4}  NER文件夹：基于规则与条件随机场混合模式的命名实体识别模块<br>
	 |-main.py：命名实体识别模块入口，包括模型训练、测试及预测<br>
	 |-model.py：NER模型算法，已实现了CRF/BiLSTM-CRF/BiLSTM+CRF+pattern<br>
	 |-data_manager.py：定义数据集输入细节<br>
	 |-utils.py：生成数据集及评测模块<br>
	 |-data文件夹：存储数据集文件<br>
	 |-models文件夹：存储配置及模型参数文件<br>
6)  static文件夹：存放css和js，HTML样式和效果文件，以及数据库全实体关系json文件<br>
7)  templates文件夹：HTML页面<br>
     |-index.html 欢迎界面<br> 
     |-search.html 搜索知识关系页面<br>
     |-all_relation.html 所有知识关系页面<br>
     |-KGQA.html 知识关系问答页面<br>
8)  triple_data文件夹：存在经NER和RE模块处理后的三元组知识<br>
9)  spider文件夹：爬虫模块<br>
	 |-get_*.py 爬取数据库中实体于百科中的图片及相关信息，已储存于本文件夹下的images和json文件夹<br>
     |-show_profile.py 展示百科图片及相关信息<br>
10)  YEDDA_label_tool文件夹：标注工具，基于YEDDA开源实体标注工具，可以支持实体与关系标注<br>
11)  requirement.txt：项目所需相关库
12)  README.md：项目介绍
<hr>

## 部署步骤：<br>
1)  配置环境<br>
    - python 3.6, CUDA 10, CUDNN 7<br>
	- 根据requiement.txt安装相关库：pip install -r requirement.txt<br>
	- ~~下载LTP语言模型 http://model.scir.yunfutech.com/model/ltp_data_v3.4.0.zip ，解压后将cws与pos模型放入./KGQA/model文件夹中~~<br>
	- 解压./KGQA/model/model.zip及./BootstrappingRE/data/data.zip<br>
	- 配置JDK 8 + Neo4j 4.2.1，修改配置文件neo4j.conf为允许无密码登录 dbms.security.auth_enabled=false；dbms.default_listen_address=0.0.0.0；及允许HTTP连接<br>
2)  NER模块训练、测试、预测<br>
    - 进入./NER文件夹<br>
	- 训练：python main.py train bilstm_crf/crf <br>
	- 测试：python main.py test bilstm_crf/crf <br>
	- 预测：python main.py predict bilstm_crf/crf <br>
3)  RE模块训练、测试、预测<br>
    - 进入./BootstrappingRE文件夹<br>
    - 训练编码器：python main.py --m train_encoder --me cnn/pcnn<br>
	- 训练相似性度量网络：python main.py --m train_sn --me cnn/pnn --ms sim/tri<br>
	- 测试整体模型：python main.py --m test --me cnn/pnn --ms sim/tri<br>
	- 预测：python main.py --m predict<br>
4)  知识图谱可视化及问答系统访问<br>
    - 进入./neo_db，运行python create_graph.py，将三元组知识导入Neo4j<br>
    - 进入./文件夹，运行python app.py<br>
	- 浏览器访问localhost:5000即可<br>
