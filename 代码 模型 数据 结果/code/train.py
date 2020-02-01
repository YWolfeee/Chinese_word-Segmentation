# -*- coding:utf-8 -*-
# 说明 - 本程序为训练部分，采用了bi-lstm模型
# 首先读取数据做预处理
# 然后借用pandas.dataframe生成标准训练数据
# 从model_generate中导入模型 训练
# 20轮 准确率约为96%
import numpy as np
import pandas as pd
import model_generate
from keras.utils import np_utils
import pickle
import re
import time
from keras.models import load_model

# 超参数
word_size, maxlen = 256, 72
batch_size, epochs = 1024, 64
# times为计数器 用于调试
times = 0


# cleancite - 负责规范化数据 删除所有引号（系因最后会直接切分引号），而每句都有，数量过多;
# 没有删除更多的符号，选择在最后分词的时候处理标点


def cleancite(s):
    s = s.replace(u'“ ', '')
    s = s.replace(u'” ', '')
    s = s.replace(u'‘ ', '')
    s = s.replace(u'’ ', '')
    return s


# change_mode 将数据模式改变为模型接受的data-label模式
def change_mode(sentence):
    seg = sentence.split()
    leng = 0
    for word in seg:
        leng += len(word)
    lis = []
    lab = []
    for word in seg:
        label = []
        if len(word) == 1:
            label = "s"
        else:
            label = "b" + "m" * (len(word) - 2) + "e"
        lis.append(word)
        lab.append(label)
    lis = u''.join(lis)
    lab = u''.join(lab)
    return list(lis), list(lab)

# traslabel - 格式化label数据


def translabel(x):
    global times
    temp = tag[x].values.reshape((-1, 1))
    ttemp = list(map(lambda y: np_utils.to_categorical(y, 5), temp))
    ttemp.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    times += 1
    return np.array(ttemp)

# train_process - 负责训练，调用keras库完成


def train_process():
    model = model_generate.model_make(maxlen, chars, word_size)
    model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
              epochs=epochs, verbose=2)
    # 保存模型 对应了这个超参
    # 关于超参数的尝试见实验报告
    model.save('result/model.h5')


if __name__ == "__main__":
    # 计时部分 预估用时等 调试用
    timeseg = []
    timeseg.append(time.time())

    # 读取数据
    file = open('data/train.txt', 'rb')
    raw_traindata = file.read().decode('utf-8')
    file.close()
    tempdata = raw_traindata.split('\r\n')  # 根据换行切分
    tempdata = u''.join(list(map(cleancite, tempdata)))
    tempdata = re.split(u'[，。！？、]', tempdata)

    data, label = [], []
    for i in tempdata:
        if i:
            x = change_mode(i)
            data.append(x[0])
            label.append(x[1])

    # 调试部分
    print("number of sentence", len(data))

    # 生成pandas.dataframe, feed 后续模型
    # 本段cited 相关代码(因为不了解dataframe的机制)
    d = pd.DataFrame(index=range(len(data)))
    d['data'] = data
    d['label'] = label
    d = d[d['data'].apply(len) <= maxlen]
    d.index = range(len(d))
    tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})
    chars = []  # 统计所有字，跟每个字编号
    for i in data:
        chars.extend(i)
    chars = pd.Series(chars).value_counts()
    chars[:] = range(1, len(chars) + 1)

    # saving
    output = open('result/chars.pkl', 'wb')
    # 序列化
    pickle.dump(chars, output)
    output.close()

    timeseg.append(time.time())
    print("input time:", timeseg[1] - timeseg[0])

    # 生成相应格式
    # 本段cited 相关代码
    d['x'] = d['data'].apply(lambda x: np.array(
        list(chars[x]) + [0] * (maxlen - len(x))))
    d['y'] = d['label'].apply(translabel)

    timeseg.append(time.time())
    print("standarize", timeseg[2] - timeseg[1])

    # 开始训练
    train_process()

    timeseg.append(time.time())
    print("train", timeseg[3] - timeseg[2])
