# SEBERTNets：一种面向金融领域的事件主体抽取方法


# 简介

“事件识别”是舆情监控领域和金融领域的重要任务之一，“事件”在金融领域是投资分析，资产管理的重要决策参考。“事件识别”的复杂性在于事件类型和事件主体的判断，比如“公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”，对于“产品出现问题”事件类型，该句中事件主体是“公司A”，而不是“公司B”或“公司C”。我们称发生特定事件类型的主体成为事件主体，本任务中事件主体范围限定为：公司和机构。事件类型范围确定为：产品出现问题、高管减持、违法违规…

 
本次评测任务的主要目标是从真实的新闻语料中，抽取特定事件类型的主体。即给定一段文本T，和文本所属的事件类型S，从文本T中抽取指定事件类型S的事件主体。


输入：一段文本，事件类型S

输出：事件主体

示例：

样例1

输入：”公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”， “产品出现问题”  

输出： “公司A”

 

样例2

输入：“公司A高管涉嫌违规减持”，“交易违规”

输出： “公司A”
 
 
 # 下载数据

download dataset from
百度网盘
- ccks2019_event_entity_extract.zip

https://pan.baidu.com/s/1HNTcqWf0594rtmwBd1p9HQ
提取码：jh4u



-  event_type_entity_extract_test.csv

https://pan.baidu.com/s/1cWRq-9IKx8lOWakZFLhS-A
提取码：qdr9

or

download dataset from
Dropbox
- ccks2019_event_entity_extract.zip

https://www.dropbox.com/s/lli5mgip2clguya/ccks2019_event_entity_extract.zip?dl=0

-  event_type_entity_extract_test.csv

https://www.dropbox.com/s/e0ajdb93s2lfdw0/event_type_entity_extract_test.csv?dl=0


# 方法

“事件识别”是舆情监控领域和金融领域的重要任务之一，在金融领域是投资分析、资产管理的重要决策参考。然而，“事件识别”的复杂性复杂性在于事件类型和事件主体的判断。本文提出了一种新的模型，SequenceEnhancedBERTNetworks(简称:SEBERTNets)，该模型既可以继承BERT模型的优点，即在少量的标签样本中可以取得很好的很好的效果，同时利用序列模型(如：GRU、LSTM)可以捕捉文本的序列语义信息。


# 运行环境

- tensorflow==1.14.0 

- keras==2.2.4

- keras-bert==0.89.0

- scikit-learn==0.24.2 

- pandas==1.1.5  

- tqdm==4.64.0 


# 运行方法

```shell
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
python SEBERT_model.py
```


# docker上运行
```shell
sudo docker pull tensorflow/tensorflow:1.14.0-gpu-py3
sudo docker run  -dit --restart unless-stopped  --gpus=all --name=evententityextraction tensorflow/tensorflow:1.14.0-gpu-py3
sudo docker exec -it evententityextraction bash
apt update
apt install git
apt install wget
git clone https://github.com/hecongqing/CCKS2019_EventEntityExtraction_Rank5.git
cd CCKS2019_EventEntityExtraction_Rank5 
pip install -r requirements.txt
wget -P ./bert https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip -d ./bert ./bert/chinese_L-12_H-768_A-12.zip 
python ./src/SEBERT_model.py
```

