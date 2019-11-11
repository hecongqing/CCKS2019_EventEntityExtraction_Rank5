#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import gc
from random import choice


import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 




maxlen = 140 # 140
learning_rate = 5e-5 # 5e-5
min_learning_rate = 1e-5 # 1e-5


config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'

model_save_path="./"

import os
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)



token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1

#读取训练集
D = pd.read_csv('../data/ccks2019_event_entity_extract/event_type_entity_extract_train.csv', encoding='utf-8', names = ['a','b','c','d'])
D = D[D["c"] != u'其他']
classes = set(D["c"].unique())

entity_train= list(set(D['d'].values.tolist()))


# ClearData
D.drop("a", axis=1, inplace=True) # drop id
D["d"] = D["d"].map(lambda x:x.replace(u'其他',''))
D["e"] = D.apply(lambda row:1 if row[2] in row[0] else 0,axis=1)
D = D[D["e"] == 1]
#D.drop_duplicates(["b","c"],keep='last',inplace = True) # drop duplicates

train_data = []
for t,c,n in zip(D["b"], D["c"], D["d"]):   
    train_data.append((t, c, n))

D = pd.read_csv('../data/ccks2019_event_entity_extract/event_type_entity_extract_eval.csv',header=None,names=["id","text","event"])
D['event']=D['event'].map(lambda x: "公司股市异常" if x=="股市异常" else  x)
D['text']=D['text'].map(lambda x:x.replace("\x07","").replace("\x05","").replace("\x08","").replace("\x06","").replace("\x04",""))

import re
comp=re.compile(r"(\d{4}-\d{1,2}-\d{1,2})")
D['text']=D['text'].map(lambda x:re.sub(comp,"▲",x))

test_data = []
for id,t,c in zip(D["id"], D["text"], D["event"]):
    test_data.append((id, t, c))


additional_chars = set()
for d in train_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove(u'，')


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    s1[start] = 1
                    s2[end] = 1
                    x1, x2 = tokenizer.encode(first=text)
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, S1, S2], None
                        X1, X2, S1, S2 = [], [], [], []






from keras.optimizers import Optimizer
import keras.backend as K


class AccumOptimizer(Optimizer):
    """继承Optimizer类，包装原有优化器，实现梯度累积。
    # 参数
        optimizer：优化器实例，支持目前所有的keras优化器；
        steps_per_update：累积的步数。
    # 返回
        一个新的keras优化器
    Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding optimizer of gradient accumulation.
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        steps_per_update: the steps of gradient accumulation
    # Returns
        a new keras optimizer.
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(AccumOptimizer, self).__init__(**kwargs)
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.steps_per_update = steps_per_update
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.cond = K.equal(self.iterations % self.steps_per_update, 0)
            self.lr = self.optimizer.lr
            self.optimizer.lr = K.switch(self.cond, self.optimizer.lr, 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, K.switch(self.cond, value, 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
            # 覆盖原有的获取梯度方法，指向累积梯度
            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss, params):
                return [ag / self.steps_per_update for ag in self.accum_grads]
            self.optimizer.get_gradients = get_gradients
    def get_updates(self, loss, params):
        self.updates = [
            K.update_add(self.iterations, 1),
            K.update_add(self.optimizer.iterations, K.cast(self.cond, 'int64')),
        ]
        # 累积梯度 (gradient accumulation)
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        grads = self.get_gradients(loss, params)
        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(K.update(ag, K.switch(self.cond, ag * 0, ag + g)))
        # 继承optimizer的更新 (inheriting updates of original optimizer)
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates
    def get_config(self):
        iterations = K.eval(self.iterations)
        K.set_value(self.iterations, 0)
        config = self.optimizer.get_config()
        K.set_value(self.iterations, iterations)
        return config

#定义模型

import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.engine.topology import Layer
from keras.models import Model
from keras.callbacks import *
from keras.optimizers import Adam,SGD
from sklearn.model_selection import KFold


def modify_bert_model_3(): # BiGRU + DNN # 
    # [0.8855585831063352, 0.878065395095436, 0.8739782016349456, 0.8773841961853542,
    #  0.8827539195638037, 0.8766189502386503, 0.8684389911384458, 0.8663940013633947, 
    #  0.8663940013633947, 0.8718473074301977]
    # mean score: 0.8747433547119957
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）

    x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
    x = bert_model([x1, x2])

    l = Lambda(lambda t: t[:, -1])(x)
    x = Add()([x, l])
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = Dense(1024, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, use_bias=False, activation='tanh')(x)

    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    model = Model([x1_in, x2_in], [ps1, ps2])

    train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
    ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
    loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    train_model.compile(optimizer=Adam(learning_rate),metrics=['accuracy'])
    train_model.summary()
    return model, train_model




def modify_bert_model_h3(): # BiGRU + DNN # 
    # [0.8855585831063352, 0.878065395095436, 0.8739782016349456, 0.8773841961853542,
    #  0.8827539195638037, 0.8766189502386503, 0.8684389911384458, 0.8663940013633947, 
    #  0.8663940013633947, 0.8718473074301977]
    # mean score: 0.8747433547119957
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))  # 待识别句子输入
    x2_in = Input(shape=(None,))  # 待识别句子输入
    s1_in = Input(shape=(None,))  # 实体左边界（标签）
    s2_in = Input(shape=(None,))  # 实体右边界（标签）

    x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
    x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
    x = bert_model([x1, x2])

    l = Lambda(lambda t: t[:, -1])(x)
    x = Add()([x, l])
    x = Dropout(0.1)(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
    x = Bidirectional(CuDNNGRU(200, return_sequences=True))(x)
    x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

    x = Dense(1024, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, use_bias=False, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(8, use_bias=False, activation='tanh')(x)

    ps1 = Dense(1, use_bias=False)(x)
    ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
    ps2 = Dense(1, use_bias=False)(x)
    ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

    model = Model([x1_in, x2_in], [ps1, ps2])

    train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

    loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
    ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
    loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
    loss = loss1 + loss2

    train_model.add_loss(loss)
    sgd=SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    train_model.compile(optimizer=sgd,metrics=['accuracy'])
    train_model.summary()
    return model, train_model





def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2  = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end+1].argmax() + start
    a = text_in[start-1: end]
    return a

class Evaluate(Callback):
    def __init__(self, dev_data, model_path):
        self.ACC = []
        self.best = 0.
        self.passed = 0
        self.dev_data = dev_data
        self.model_path = model_path
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            print("save best model weights ...")
            train_model.save_weights(self.model_path)
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        # F = open('dev_pred.json', 'w', encoding = 'utf-8')
        for d in tqdm(iter(self.dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            # s = ', '.join(d + (R,))
            # F.write(s + "\n")
        # F.close()
        return A / len(self.dev_data)

def test(test_data,result_path):
    F = open(result_path, 'w', encoding = 'utf-8')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1], d[2]))
        # s = s.encode('utf-8')
        F.write(s)
    F.close()


def evaluate(dev_data):
    A = 1e-10
    # F = open('dev_pred.json', 'w', encoding='utf-8')
    for d in tqdm(iter(dev_data)):
        R = extract_entity(d[0], d[1])
        if R == d[2]:
            A += 1
    #     s = ', '.join(d + (R,))
    #     F.write(s + "\n")
    # F.close()
    return A / len(dev_data)


# Model
flodnums = 10
kf = KFold(n_splits=flodnums, shuffle=True, random_state=520).split(train_data)

score = []


for i, (train_fold, test_fold) in enumerate(kf):
    print("kFlod ",i,"/",flodnums)
    train_ = [train_data[i] for i in train_fold]
    dev_ = [train_data[i] for i in test_fold]

    #model, train_model = modify_bert_model_1()
    # model, train_model = modify_bert_model_2()
    model, train_model = modify_bert_model_3()
    # model, train_model = modify_bert_model_4()
    # model, train_model = modify_bert_model_5()
    #model, train_model = modify_bert_model_6()
    # model, train_model = modify_bert_model_7()
    # model, train_model = modify_bert_model_8()
    # model, train_model = modify_bert_model_9()

    #model, train_model = modify_bert_model_0()
    
    train_D = data_generator(train_)
    dev_D = data_generator(dev_)
    
    model_path = model_save_path+"modify_bert_model" + str(i) + ".weights"
    if not os.path.exists(model_path):
        evaluator = Evaluate(dev_, model_path)
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch = len(train_D),
                                  epochs = 5,
                                  callbacks = [evaluator],
                                  validation_data = dev_D.__iter__(),
                                  validation_steps = len(dev_D)
                                 )
        print("load best model weights ...")
        del train_model
        gc.collect()
        del model
        gc.collect()
        K.clear_session()

    model, train_model = modify_bert_model_h3()
    
    

    model_h_path = model_save_path+"modify_bert_model_h" + str(i) + ".weights"
    if not os.path.exists(model_h_path):
        train_model.load_weights(model_path)
        evaluator = Evaluate(dev_, model_h_path)
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch = len(train_D),
                                  epochs = 10,
                                  callbacks = [evaluator],
                                  validation_data = dev_D.__iter__(),
                                  validation_steps = len(dev_D)
                                 )
        score.append(evaluate(dev_))
        print("valid evluation:", score)
        print("valid mean score:", np.mean(score))

    train_model.load_weights(model_h_path)





    result_path = model_save_path+"result_k" + str(i) + ".txt"
    test(test_data,result_path)

    del train_model
    gc.collect()
    del model
    gc.collect()
    K.clear_session()


# Name           Score            Models
# (1)     -->   0.90182    Last + BiLSTM + TD + DNN3
# (2)     -->   0.90046    Last + DNN3
# (3)     -->   0.90585    Last + BiGRU2 + DNN3
# (4)     -->   0.89549    Baseline
# (5)     -->   0.90121    CNN + BiLSTM + TD
# (6)     -->   0.90447    GRU2 + TD + CNN
# (7)     -->   0.89835    Last + BiGRU2 + DNN2
# (8)     -->   0.90179    BiGRU2 + Last + DNN3
# (9)     -->   0.90099    Last + BiGRU2 + TD + DNN3

# (0)     -->   

# (1,4)   -->   0.90200  
# (1,3)   -->   0.90254
# (1,2)   -->   0.90239

# (3,5)   -->   0.90361
# (3,6)   -->   0.90500




####### Submit #######
data = pd.DataFrame(columns=["sid","company"])

dataid = pd.read_csv(model_save_path+"result_k0.txt",sep=',',names=["sid","company"])[['sid']]

for i in range(flodnums):
    datak = pd.read_csv(model_save_path+"result_k"+str(i)+".txt",sep=',',names=["sid","company"])
    print(datak.shape)
    data = pd.concat([data, datak], axis = 0)

submit = data.groupby(['sid','company'],as_index=False)['sid'].agg({"count":"count"})

print(submit.shape)
print(submit[submit.company == 'NaN'])

submit = submit.sort_values(by=["sid","count"],ascending=False).groupby("sid",as_index=False).first()

print(submit.shape)

submit = dataid.merge(submit,how='left',on = 'sid').fillna("NaN")
print(data[['sid']].drop_duplicates().shape)
print(submit.shape)

submit[['sid','company']].to_csv(model_save_path+"result.txt",header=None,index=False,sep=',')

print(submit)
