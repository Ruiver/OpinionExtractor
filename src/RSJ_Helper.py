import numpy as np
import json
from sklearn import preprocessing
from RSE import SeqLabeling
from Arguments import Arguments
from Helper import *
from utils import *
import re

class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        e1s = []
        chars = []
        e2s = []
        targets = []
        cut_features = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            char, e1, e2, cut_feature, target = line
            padding = [0] * (max_length - len(char))
            chars.append(char + padding)
            e1s.append(e1)
            e2s.append(e2)
            cut_features.append(cut_feature)
            targets.append(target)
        return [chars, e1s,e2s, cut_features, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

class RSJ_Helper(object):
    def __init__(self):
        args = Arguments('RSE')
        # ner : 命名实体识别模型
        self.ner = SeqLabeling(args)
        #
        self.ner.restore_model()
        self.embeding = self.ner.get_embedding()
        self.fixlen = 70
        self.char_id = self.ner.char_to_id
        self.tag_id = {'available':0,'none':1}
        self.id_tag = {t: i for i, t in self.tag_id.items()}

    @staticmethod
    def pos_embed(x):
        if x < -60:
            return 0
        if -60 <= x <= 60:
            return x + 61
        if x > 60:
            return 122

    @staticmethod
    def get_cut_feature(sentence,e1,e2):
        """处理第二个实体在第一个实体之间的问题"""
        if e1['start']>e2['end']:
            sub = sentence[e2['end']:e1['start']]
        else:
            sub = sentence[e1['end']:e2['start']]
        """如果两个实体间跨句了，就返回3"""
        t = re.findall("([.。\"\'“”‘’?!])", sub)
        if t:
            return 3
        r = re.findall(u"([,，\$\s])",sub)
        # print(sub,t,r)
        return 2 if len(r)>1 else len(r)

    def load_data(self, file):
        with open(file, encoding='utf8') as r:
            data = []

            for line in r.read().split('\n\n'):
                if not line:
                    continue
                sentence, entities, rels = line.split('\n')
                entities = json.loads(entities)
                rels = json.loads(rels)
                # sentence_id,_,self.embeding,lstm_outputs = self.ner.evaluate_line(sentence)
                for rel in rels:
                    if len(sentence) < self.fixlen:
                        sentence_ = list(sentence) + ['<PAD>'] * (self.fixlen - len(sentence))
                    else:
                        sentence_ = list(sentence)[:self.fixlen]
                    # length of sentence is 70
                    # fixlen = 70
                    # max length of position embedding is 60 (-60~+60)
                    maxlen = 60
                    words = []
                    e1s = []
                    e2s = []
                    for i in range(self.fixlen):
                        words.append(
                            self.char_id[sentence_[i]] if sentence_[i] in self.char_id.keys() else self.char_id[
                                '<UNK>'])
                        e1s.append(RSJ_Helper.pos_embed(i - entities[int(rel['start'])]['start']))# 实体1的起始位置
                        e2s.append(RSJ_Helper.pos_embed(i - entities[int(rel['end'])]['start'])) # 实体2的起始位置
                    cut_feature = RSJ_Helper.get_cut_feature(sentence,entities[int(rel['start'])],entities[int(rel['end'])])
                    print(cut_feature)
                    data.append([words, e1s, e2s, cut_feature, self.tag_id[rel['rel_type']]])
        return data

    def load_train(self,file):
        self.train_data = self.load_data(file)

    def load_dev(self,file):
        self.dev_data = self.load_data(file)

    def load_test(self,file):
        self.test_data = self.load_data(file)

    def create_batch(self,batch_size):
        self.train_batch = BatchManager(self.train_data, batch_size)
        self.dev_batch = BatchManager(self.dev_data, 100)
        self.test_batch = BatchManager(self.test_data, 100)

    def input_from_line(self,sentence,interal=10):
        data = []
        """
        # sentence_id是切词序列，entities实体列表，每个元素是一个字典，
        包括value，start，end，和type四个属性embedding NER的词向量，LSTM——output NER中BiLSTM的隐层输出
        """
        sentence_id, entities, embeding, lstm_outputs = self.ner.evaluate_line(sentence)
        """# 对取出来的实体进行配对，rels_matrix是一个二维的实体关系矩阵。"""
        rels_matrix = rel_extract(entities,['pair-feature','pair-perspective'],interal,False)
        """# 对矩阵中存在可能配对关系的地方，对称矩阵"""
        xs, ys = np.where(rels_matrix > 0)
        # print(xs, ys)
        """# 若xs为空则返回{"label":False,'entities':entities}"""
        if len(xs) == 0:
            return {"label":False,'entities':entities}
        """# 若不为空，则提取这些潜在的配对实体。"""
        rels = []
        for x, y in zip(xs, ys):
            """# 对矩阵中的每一个配对的元素对进行提取，rel是一个配对实例，包括两个实体在实体序列中的位置，和配对的值e1_e2形式，
            # 关系初始化为none"""
            rel = {}
            rel['start'] = str(x)
            rel['end'] = str(y)
            rel['value'] = entities[x]['value'] + '_' + entities[y]['value']
            rel['rel_type'] = "none"
            rels.append(rel)
        for rel in rels:
            """# 对提取到的每一处潜在配对关系进行判断"""
            # e1_start = entities[int(rel['start'])]['start']
            # e1_end = entities[int(rel['start'])]['end']
            # e2_start = entities[int(rel['end'])]['start']
            # e2_end = entities[int(rel['end'])]['end']
            # e1_emb = np.max(lstm_outputs[0, e1_start:e1_end], 0)
            # e2_emb = np.max(lstm_outputs[0, e2_start:e2_end], 0)
            if len(sentence) < self.fixlen:
                sentence_ = list(sentence) + ['<PAD>'] * (self.fixlen - len(sentence))
            else:
                sentence_ = list(sentence)[:self.fixlen]
            words = []
            e1s = []
            e2s = []
            for i in range(self.fixlen):
                """输入转id"""
                words.append(
                    self.char_id[sentence_[i]] if sentence_[i] in self.char_id.keys() else self.char_id['<UNK>'])
                """"""
                """这里是，对每一对潜在配对的实体的实体1的起始位置和实体2的结束位置，的位置embedding"""
                e1s.append(RSJ_Helper.pos_embed(i - entities[int(rel['start'])]['start']))
                e2s.append(RSJ_Helper.pos_embed(i - entities[int(rel['end'])]['start']))
            """获取切句特征，cut_feature = 0,1,2,3"""
            cut_feature = RSJ_Helper.get_cut_feature(sentence,entities[int(rel['start'])],entities[int(rel['end'])])
            # sentence_emb = [self.embeding[w_id] for w_id in sentence_id]
            print(rel, cut_feature)
            data.append([words, e1s, e2s, cut_feature, self.tag_id[rel['rel_type']]])
        return {"label":True,'entities':entities, 'rels':rels,'rsj_data':BatchManager.pad_data(data)}


if __name__=='__main__':
    # args = Arguments()
    # ner = SeqLabeling(args)
    # ner.restore_model()
    # print(ner.evaluate_line('手机不错'))
    lb = preprocessing.LabelBinarizer()
    lb.fit(['positive', 'negative', 'non'])
    print(lb.inverse_transform([1,0,0]))