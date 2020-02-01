import pandas as pd
import re
import numpy as np
import pickle
import keras.models as mod
import keras.layers as lay
#from google.colab import drive
# drive.mount("/content/gdrive")

# 线上跑模型使用，无影响
uploadpath = '/content/gdrive/My Drive/upload/'

# maxlen本来应该是64的，但是竟然有一句话有66个汉字而且没标点....
word_size, maxlen = 256, 72

# 维特比算法 动态规划求最大概率路径


def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 开始

    for eachlayer in range(1, len(nodes)):  # 后面的每一层
        prevpaths = paths.copy()  # 保存上一层
        paths = {}  # clear

        for node in nodes[eachlayer].keys():
            # 求本层最短路
            nextfind = {}
            # 考虑上一层的所有节点和本层的所有节点
            for last_path in prevpaths.keys():
                if last_path[-1] + node in trans.keys():  # 非0
                    nextfind[last_path + node] = prevpaths[last_path] + nodes[eachlayer][node] + trans[
                        last_path[-1] + node]
            minpath = pd.Series(nextfind).sort_values()   # 最短路径
            node_subpath = minpath.index[-1]
            node_probability = minpath[-1]

            # 把 node 的最短路径添加到 paths 中
            paths[node_subpath] = node_probability
    # finish road calculating
    return pd.Series(paths).sort_values().index[-1]


def cut_one(phrases):
    if phrases:  # 非空
        prearray = np.array([list(chars[list(phrases)].fillna(
            0).astype(int)) + [0] * (maxlen - len(phrases))])
        r = np.log(model.predict(prearray,
                                 verbose=False)[
            0][:len(phrases)])
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        outputsequence = viterbi(nodes)
        segresult = []
        for i in range(len(phrases)):
            if outputsequence[i] in ['s', 'b']:
                segresult.append(phrases[i])
            else:
                segresult[-1] += phrases[i]
        return segresult
    else:
        return []


# 强制性分割
force_cut = re.compile(r'([\da-zA-Z ]+)|[“”‘’""《》『』。，%：;；、？！\.\?,!]')


def cut_sentence(sentence):
    global force_cut
    result = []
    iter = 0
    for i in force_cut.finditer(sentence):
        result.extend(cut_one(sentence[iter:i.start()]))
        result.append(sentence[i.start():i.end()])
        iter = i.end()
    result.extend(cut_one(sentence[iter:]))
    return result


if __name__ == "__main__":
    file = open('result/chars.pkl', 'rb')
    chars = pickle.load(file)
    file.close()

    # 加载模型
    model = mod.load_model('result/model.h5')

    # 初始化转移概率
    trans = {'be': 0.5,
             'bm': 0.5,
             'eb': 0.5,
             'es': 0.5,
             'me': 0.5,
             'mm': 0.5,
             'sb': 0.5,
             'ss': 0.5
             }
    trans = {i: np.log(trans[i]) for i in trans.keys()}
    inputfile = open('data/test.txt', 'rb')
    raw_traindata = inputfile.read().decode('utf-8')
    readdata = raw_traindata.split('\r\n')
    inputfile.close()

    # 输出结果
    writer = open("data/myanswer.txt", "w")
    for sentence in readdata:
        print(sentence)
        answer = ' '.join(cut_sentence(sentence)) + "\n"
        writer.write(answer)
    print("finish")
    writer.close()
