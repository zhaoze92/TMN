# coding: utf-8
'''
训练数据预处理：

'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import numpy as np
import gensim
import os
import sys
from scipy import sparse
import pickle
import json
import jieba

# 获取停用词表
from gensim.parsing.preprocessing import STOPWORDS
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

# if len(sys.argv) != 2:
#     print("Usage:\npython process_tmn.py <input_data_file>")
#     exit(0)
#data_file = sys.argv[1]

# 训练数据文件读取.文件格式： query######label
data_file = "../data/multiCluster/mlp_data.txt"
data_dir = os.path.dirname(data_file)

# 编码格式处理
with open(os.path.join(data_file), 'U') as fin:
    text = gensim.utils.any2utf8(fin.read(), 'utf8').strip()
# 按行分割
news_lst = text.split("\n")
print(news_lst[:5])

msgs = [] # 保存所有query
labels = []  # 保存所有样本的label id
label_dict = {} # 保存label的id映射

for n_i, line in enumerate(news_lst):
    msg, label = line.strip().split("\t")
    # msg = list(gensim.utils.tokenize(msg, lower=True))
    # 分词
    msg = list(jieba.cut(msg, cut_all=False))
    msgs.append(msg)
    if label not in label_dict:
        label_dict[label] = len(label_dict)
    labels.append(label_dict[label])
print("read done.")

# 建立语料特征（此处即是word）的索引字典
dictionary = gensim.corpora.Dictionary(msgs)
print("build dictionary done.")

# 重新拷贝一份用于处理
import copy
bow_dictionary = copy.deepcopy(dictionary)

# 去除停用词
stopWordsPath = "../data/multiCluster/stop_words_ch.txt"
stopWords = []
for word in open(stopWordsPath).readlines():
    stopWords.append(word.strip())
# bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, STOPWORDS)))
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, stopWords)))

# 过滤长度为1的单词
len_1_words = list(filter(lambda w: len(w.decode("utf8")) == 1, bow_dictionary.values()))
bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
print("filter stopwords and 1s length done.")

# 1.去掉出现次数低于no_below的; 2.去掉出现次数高于no_above的。注意这个小数指的是百分数; 3.在1和2的基础上，保留出现频率前keep_n的单词
bow_dictionary.filter_extremes(no_below=3, keep_n=None)

# 为所有单词分配新的单词ID，缩小任何间隙。
bow_dictionary.compactify()
print("compactify done")

# 参数: msgs, dictionary, bow_dictionary, labels
def get_wids(text_doc, seq_dictionary, bow_dictionary, ori_labels):
    seq_doc = []

    # build bow
    row = [] # 统计行数，过滤之后保留的
    col = [] # 统计列的的取值
    value = []
    row_id = 0
    m_labels = []

    # 迭代所有的query
    for d_i, doc in enumerate(text_doc):
        tmp = bow_dictionary.doc2bow(doc)
        # 如果query在处理之后太短，过滤掉
        # Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)
        if len(bow_dictionary.doc2bow(doc)) < 3:    # filter too short
            continue
        for i, j in bow_dictionary.doc2bow(doc):
            row.append(row_id)
            col.append(i)
            value.append(j)
        row_id += 1

        wids = list(map(seq_dictionary.token2id.get, doc))
        wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
        m_labels.append(ori_labels[d_i])
        seq_doc.append(wids) # 存储doc的term id。id做了加1处理。

    # 要多使用 list(map(func, nums)) 这种格式的参数。
    lens = list(map(len, seq_doc)) # 获取每个句子的长度。

    # 经过上面的过滤之后重新生成稀疏矩阵。coo_matrix函数含义参见函数定义即可。row表示value的值依次在第几行，col表示value的值依次在第几列。
    bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))

    logging.info("get %d docs, avg len: %d, max len: %d" % (len(seq_doc), np.mean(lens), np.max(lens)))
    return seq_doc, bow_doc, m_labels


seq_title, bow_title, label_title = get_wids(msgs, dictionary, bow_dictionary, labels)
print("get wids done.")

# shuf data
indices = np.arange(len(seq_title))
np.random.shuffle(indices)
seq_title = np.array(seq_title)[indices]

# 划分训练集和测试集
nb_test_samples = int(0.2 * len(seq_title))
seq_title_train = seq_title[:-nb_test_samples]
seq_title_test = seq_title[-nb_test_samples:]


bow_title = bow_title.tocsr()
bow_title = bow_title[indices]
bow_title_train = bow_title[:-nb_test_samples]
bow_title_test = bow_title[-nb_test_samples:]

label_title = np.array(label_title)[indices]
label_title_train = label_title[:-nb_test_samples]
label_title_test = label_title[-nb_test_samples:]
print("gen sample done.")


# save
logging.info("save data...")
pickle.dump(seq_title, open(os.path.join(data_dir, "dataMsg"), "wb"))
pickle.dump(seq_title_train, open(os.path.join(data_dir, "dataMsgTrain"), "wb"))
pickle.dump(seq_title_test, open(os.path.join(data_dir, "dataMsgTest"), "wb"))
pickle.dump(bow_title, open(os.path.join(data_dir, "dataMsgBow"), "wb"))
pickle.dump(bow_title_train, open(os.path.join(data_dir, "dataMsgBowTrain"), "wb"))
pickle.dump(bow_title_test, open(os.path.join(data_dir, "dataMsgBowTest"), "wb"))
pickle.dump(label_title, open(os.path.join(data_dir, "dataMsgLabel"), "wb"))
pickle.dump(label_title_train, open(os.path.join(data_dir, "dataMsgLabelTrain"), "wb"))
pickle.dump(label_title_test, open(os.path.join(data_dir, "dataMsgLabelTest"), "wb"))
dictionary.save(os.path.join(data_dir, "dataDictSeq"))
bow_dictionary.save(os.path.join(data_dir, "dataDictBow"))
json.dump(label_dict, open(os.path.join(data_dir, "labelDict.json"), "w"), indent=4)
logging.info("save done!")
