# CCKS2019EventEntityExtraction_Rank5
SEBERTNets：一种面向金融领域的事件主体抽取方法

# 简介



# 方法

“事件识别”是舆情监控领域和金融领域的重要任务之一，在金融领域是投资分析、资产管理的重要决策参考。然而，“事件识别”的复杂性复杂性在于事件类型和事件主体的判断。本文提出了一种新的模型，SequenceEnhancedBERTNetworks(简称:SEBERTNets)，该模型既可以继承BERT模型的优点，即在少量的标签样本中可以取得很好的很好的效果，同时利用序列模型(如：GRU、LSTM)可以捕捉文本的序列语义信息。


# 运行环境

- tensorflow==1.14.0 

- keras==2.2.4

- keras-bert==0.69.0

# 运行方法

```shell
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
python SEBERT_model.py
```


