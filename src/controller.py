from Arguments import Arguments
from RSJ import RSJ
from aip import AipNlp
from bosonnlp import BosonNLP
from Helper import *
import pandas as pd
import numpy as np
import os
import jieba
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def stop_words():
    with codecs.open('./stopword.txt', 'r') as f:
        stop_list = [e[:-1] for e in f.readlines()]
        return stop_list
stop = stop_words()
class Controller:
    def __init__(self):
        args = Arguments('RSJ')
        self.rsj = RSJ(args)
        self.rsj.restore_model()
        APP_ID = '14465679'
        API_KEY = 'DDNA68lRaVxKCUHP13t79acC'
        SECRET_KEY = 'RisCmApExjn5hcSH0KHul71Uldza8vDe'
        self.feature_maps = {}
        with open('../data/feature_maps.txt',encoding='utf8') as r:
            for line in r:
                features = line.split(' ')
                self.feature_maps[features[0]] = features

        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        self.boson = BosonNLP('m9YSebz-.27886.-Jh0KNhk__Q2')

    def processing(self, sentence):
        rel = self.rsj.evaluate_line(sentence)
        result = {'sentence':sentence,'entities':[],'rels':[]}
        sub_sentences,indexes = DataUtils.cut_to_sentence(sentence,index=True)
        def sub(indexes,index):
            index = int(index)
            for i,v in enumerate(indexes):
                if v[0]<=index and v[1]>index:
                    return i
        def boson_sen(boson,tag):
            r = 1
            if (boson[0]-boson[1])>0.3:
                r=2
            elif (boson[1]-boson[0])>0.3:
                r=0
            return tag[r]

        ps = []
        fs = []
        for item in rel['rels']:
            if item['rel_type']=='none':
                continue
            if rel['entities'][int(item['start'])]['type'] == 'pair-perspective':
                ps.append(int(item['start']))
                fs.append(int(item['end']))
            else:
                ps.append(int(item['end']))
                fs.append(int(item['start']))
        id2sen = {0:'negative',1:'neutral',2:'positive'}

        for i,item in enumerate(rel['entities']):
            # 这个for循环判断复杂观点词和整体观点词
            item_ = item
            if 'perspective' in item['type'] and i not in ps:
                # 整体观点词
                # item_['type'] = 'perspective'
                index = sub(indexes, item['start'])
                feature_value = '手机'
                perspective_value = item['value']
                sentiment_resu = self.boson.sentiment(sub_sentences[index])[0]
                sentiment = boson_sen(sentiment_resu,id2sen)
                # sentiment_resu = self.client.sentimentClassify(sub_sentences[index])
                # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]
                # if sentiment_resu['items'][0]['confidence']<0.6:
                #     sentiment = 'positive'
                result['rels'].append({'feature':-1,'perspective':i,'sentiment':sentiment,'parent_feature':'手机',
                                       'value':'{}-{}'.format(feature_value,perspective_value),'feature_value':feature_value,
                                       'perspective_value':perspective_value})
            elif 'feature' in item['type'] and i not in fs:
                # item_['type'] = 'feature'
                index = sub(indexes, item['start'])
                sentiment_resu = self.boson.sentiment(sub_sentences[index])[0]
                sentiment = boson_sen(sentiment_resu,id2sen)

                # sentiment_resu = self.client.sentimentClassify(sub_sentences[index])
                # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]
                feature_value = item['value']
                perspective_value = sentiment
                nums = 0
                parent_feature = "其他"
                for name, features in self.feature_maps.items():
                    if feature_value in features and len(features) > nums:
                        parent_feature = name
                        nums = len(features)
                result['rels'].append(
                    {'feature': i, 'perspective': -1, 'sentiment': sentiment, 'parent_feature': parent_feature,
                     'value': '{}-{}'.format(feature_value, perspective_value),'feature_value':feature_value,'perspective_value':perspective_value})
            # elif item['type'] == 'pair-perspective':
            #     item_['type'] = 'perspective'
            # else:
            #     item_['type'] = 'feature'
            result['entities'].append(item_)
        for item in rel['rels']:
            # 判断词对
            if item['rel_type']=='none':
                continue
            if 'perspective' in result['entities'][int(item['start'])]['type']:
                perspective = int(item['start'])
                feature = int(item['end'])
            else:
                perspective = int(item['end'])
                feature = int(item['start'])
            index1 = sub(indexes, rel['entities'][int(item['start'])]['start'])
            index2 = sub(indexes, rel['entities'][int(item['start'])]['end'])
            feature_value = result['entities'][int(feature)]['value']
            perspective_value = result['entities'][perspective]['value']
            sentiment_resu = self.boson.sentiment(''.join(sub_sentences[index1:index2+1]))[0]
            sentiment = boson_sen(sentiment_resu, id2sen)

            # sentiment_resu = self.client.sentimentClassify(''.join(sub_sentences[index1:index2+1]))
            # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]
            nums = 0
            parent_feature = '其他'
            for name, features in self.feature_maps.items():
                if feature_value in features and len(features) > nums:
                    parent_feature = name
                    nums = len(features)
            result['rels'].append({'feature':feature,'perspective':perspective,'sentiment':sentiment,
                                   'parent_feature':parent_feature,'value':'{}-{}'.format(feature_value,perspective_value),
                                   'feature_value':feature_value,'perspective_value':perspective_value})
        return result
############################################################################################################
#
#                                      修改区
#
#########################################################################################################
class Extractor:
    def __init__(self):
        # 加载特征词典
        args = Arguments('RSJ')
        self.rsj = RSJ(args)
        self.rsj.restore_model()
        #APP_ID = '14465679'
        #API_KEY = 'DDNA68lRaVxKCUHP13t79acC'
        #SECRET_KEY = 'RisCmApExjn5hcSH0KHul71Uldza8vDe'
        self.feature_maps = {}
        with open('../data/feature_maps.txt',encoding='utf8') as r:
            for line in r:
                features = line.split(' ')
                self.feature_maps[features[0]] = features

        #self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        #self.boson = BosonNLP('m9YSebz-.27886.-Jh0KNhk__Q2')

    def processing(self, sentence):
        rel = self.rsj.evaluate_line(sentence)
        #print("预设断点rel是：")
        #print(rel)
        result = {'sentence':sentence,'entities':[],'rels':[]}
        sub_sentences,indexes = DataUtils.cut_to_sentence(sentence,index=True)
        #print("预设断点sub_sentences是：")
        #print(sub_sentences)
        #print("预设断点，indexes是：")
        #print(indexes)
        def sub(indexes,index):
            index = int(index)
            for i,v in enumerate(indexes):
                if v[0]<=index and v[1]>index:
                    return i
        def boson_sen(boson,tag):
            r = 1
            if (boson[0]-boson[1])>0.3:
                r=2
            elif (boson[1]-boson[0])>0.3:
                r=0
            return tag[r]
        def clean_word(text):
            return re.sub("[^\u4e00-\u9fa5]", '', str(text).lstrip())
        ps = []
        fs = []
        for item in rel['rels']:
            if item['rel_type']=='none':
                continue
            # 对rel中的词对匹配结果进行处理，将词对索引中的特征实体的下标赋给fs将观点词赋给ps一一对应
            if rel['entities'][int(item['start'])]['type'] == 'pair-perspective':
                ps.append(int(item['start']))
                fs.append(int(item['end']))
            else:
                ps.append(int(item['end']))
                fs.append(int(item['start']))
        id2sen = {0:'negative',1:'neutral',2:'positive'}

        # 对非词对类型的观点进行处理
        for i, item in enumerate(rel['entities']):
            # rel['entities']是一个列表，里面存储着词对，观点词（对整个手机的全面评价），复杂特征（。。还很流畅之类的需要情感分析）
            # 主要有pair-feature,pair-perspective, overoll-perspective,complex-feature
            item_ = item
            # 处理overoll-perspective，和pair-perspective
            if 'perspective' in item['type'] and i not in ps:
                # item_['type'] = 'perspective'

                index = sub(indexes,item['start']) # 返回整体观点词的所在句子下标相对于sub_sentence
                feature_value = '手机'# 给它补上特征词“手机”
                perspective_value = clean_word(item['value']) # 给它补上情感词
                opinion_mention_1 = sub_sentences[index]
                #sentiment_resu = self.boson.sentiment(sub_sentences[index])[0] # 情感极性由整个句子情感分析得到
                #sentiment = boson_sen(sentiment_resu,id2sen) #给它情感极性
                # sentiment_resu = self.client.sentimentClassify(sub_sentences[index])
                # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]
                # if sentiment_resu['items'][0]['confidence']<0.6:
                #     sentiment = 'positive'
                #feature:-1 表示是整体特征“手机”i表示观点词在提取出来的实体中的位置
                # result['rels'].append({'feature':-1,'perspective':i,'sentiment':sentiment,'parent_feature':'手机',
                #                        'value':'{}-{}'.format(feature_value,perspective_value),'feature_value':feature_value,
                #                        'perspective_value':perspective_value})
                result['rels'].append({'type':1, 'feature': -1, 'perspective': i,  'parent_feature': '手机','opinion_mention':opinion_mention_1,
                                       'feature_value': feature_value,
                                       'perspective_value': perspective_value})
            # 处理complex-feature，和overall-feature类型
            elif 'feature' in item['type'] and i not in fs:
                # item_['type'] = 'feature'
                # 返回complex-feature在句子中的子句下标
                index = sub(indexes, item['start'])
                opinion_mention_2 = clean_word(sub_sentences[index])
                # 一样对这句话做情感分析
                #sentiment_resu = self.boson.sentiment(sub_sentences[index])[0]
                #sentiment = boson_sen(sentiment_resu,id2sen)

                # sentiment_resu = self.client.sentimentClassify(sub_sentences[index])
                # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]

                feature_value = clean_word(item['value'])# 特征词有
                perspective_value = opinion_mention_2# 这种情，观点词就是这句话吧
                nums = 0
                parent_feature = "其他" # 来对特征词进行判断，默认为其他
                for name, features in self.feature_maps.items(): # 遍历feature-map
                    if feature_value in features and len(features) > nums: # 如果特征值在feature列表中，
                        parent_feature = name #则让父级特征为featur_emap中的feature大类
                        nums = len(features)
                result['rels'].append(
                    {'type':2, 'feature': i, 'perspective': -1,'parent_feature': parent_feature, 'opinion_mention': opinion_mention_2,
                     'feature_value':feature_value,'perspective_value':perspective_value})
            # elif item['type'] == 'pair-perspective':
            #     item_['type'] = 'perspective'
            # else:
            #     item_['type'] = 'feature'
            result['entities'].append(item_) # 已经处理过的entities都加入result['entities']
        # 处理词对观点词
        for item in rel['rels']:
            if item['rel_type']=='none': # 对于不能构成词对的，直接跳过
                continue
            if 'perspective' in result['entities'][int(item['start'])]['type']:
                perspective = int(item['start']) # 修改观点词下标
                feature = int(item['end']) # 修改特征词的下标
            else:
                perspective = int(item['end']) # 修改观点词下标
                feature = int(item['start']) # 修改特征词下标
            index1 = sub(indexes, rel['entities'][int(item['start'])]['start'])
            index2 = sub(indexes, rel['entities'][int(item['start'])]['end'])
            feature_value = clean_word(result['entities'][int(feature)]['value']) # 修改特征词
            perspective_value = clean_word(result['entities'][perspective]['value']) # 修改观点词
            opinion_mention_3 = clean_word(''.join(sub_sentences[index1:index2+1]))
            #sentiment_resu = self.boson.sentiment(''.join(sub_sentences[index1:index2+1]))[0]
            #sentiment = boson_sen(sentiment_resu, id2sen)

            # sentiment_resu = self.client.sentimentClassify(''.join(sub_sentences[index1:index2+1]))
            # sentiment = id2sen[sentiment_resu['items'][0]['sentiment']]
            nums = 0
            parent_feature = '其他'
            for name, features in self.feature_maps.items():
                if feature_value in features and len(features) > nums:
                    parent_feature = name
                    nums = len(features)
            result['rels'].append({'type':3, 'feature':feature,'perspective':perspective,'parent_feature':parent_feature,'opinion_mention':opinion_mention_3,
                                   'feature_value': feature_value,'perspective_value':perspective_value})
        return result

def parse_result(result_dict):
    result = {}
    entities_result = [] # entities_result列表是一个提取出来的实体词列表，每一个元素是一个元祖(value, start, end, type)
    opinion_result = [] # opinion_result列表是一个提取出来的属性观点词列表，每一个元素是一个元祖(parent_feature,feature_value,perspective_value,opinion_mention)
    # 先提取实体
    for item in result_dict["entities"]:
        entities_result.append((item['value'],item['start'],item['end'],item['type']))
    for item in result_dict["rels"]:
        opinion_result.append((item["type"],item['parent_feature'],item['feature_value'], item['perspective_value'], item['opinion_mention']))
    result["entities"] = entities_result
    result["opinion"] = opinion_result
    return result

def extract_one_file(filepath,savepath,JD=True):
    extractor = Extractor()
    df = pd.read_csv(filepath, encoding='utf-8')
    def quchong(sen, thresh=2):
        l = len(sen)
        a = [0]
        for i in range(1, l):
            if sen[i] == sen[i - 1]:
                a.append(a[i - 1] + 1)
            else:
                a.append(0)
        text = ""
        for i in range(l):
            if a[i] < thresh:
                text = text + sen[i]
        return text.lstrip()
    def truncat_text(text):
        lines = []
        for l in text:
            length = len(l)
            if length<30:
                lines.append(l)
            else:
                i = length / 30 if length % 30 == 0 else length/30 +1
                for j in range(int(i)):
                    start = j*30
                    end = (j+1)*30 if (j+1)*30 < length else length
                    lines.append(l[start:end])
        return lines

    def extract_sen(text):
        result_dicts = {}
        entities_result = []
        opinion_result = []
        text = text.replace(' ', '')  # 去空格
        text = quchong(text, 2)
        text = re.split('，|\,|。|！|\!|\.|？|\?', text)
        text = truncat_text(text)
        for sen in text:
            # words = jieba.lcut(sen)
            # for w in words:
            #     if w in stop:
            #         words.remove(w)
            # sen = ''.join(words)
            try:
                result_dict = extractor.processing(sen)
            except BaseException:
                result_dict = None
            if result_dict:
                result = parse_result(result_dict)
                entities_result.append(result['entities'])
                opinion_result.append(result['opinion'])
        result_dicts['entities'] = entities_result
        result_dicts['opinion'] = opinion_result
        return result_dicts
    def extract(text):
        try:
            result_dict = extractor.processing(text)
        except BaseException:
            result_dict = None
        if result_dict:
            result = parse_result(result_dict)
            return result
        else:
            return {}
    if JD:
        df["opinion"] = df.apply(lambda row: extract_sen(row["content"]),axis=1)
    else:
        df["opinion"] = df.apply(lambda row: extract_sen(row["rateContent"]), axis=1)
    df.to_csv(savepath, header=True, encoding='utf-8', index=True, mode='w')
def work():
    PHONE_BRAND = ['vivo_Z3', 'iphone_XR', 'Galaxy_note9', 'huawei_mate20', 'meizu16', 'oneplus_6T', 'OPPO_R17',
                   'xiaomi9']
    ORIGINAL_JD_FILE_PATH = r'../review_data/JD/'
    ORIGINAL_TB_FILE_PATH = r'../review_data/TB/'
    SOURCES = ['_JD', '_TB']
    FILE_POSIX = '.CSV'
    for brand in PHONE_BRAND:
        # 先提取京东的
        print("正在处理京东评论" + brand)
        filepath_jd = ORIGINAL_JD_FILE_PATH + brand + SOURCES[0] + FILE_POSIX
        savepath_jd = r'../review_data/extracted2/JD/' + brand + SOURCES[0] + FILE_POSIX
        extract_one_file(filepath_jd, savepath_jd, JD=True)
        # 再提取淘宝的
        print("正在处理淘宝评论" + brand)
        filepath_tb = ORIGINAL_TB_FILE_PATH + brand + SOURCES[1] + FILE_POSIX
        savepath_tb = r'../review_data/extracted2/TB/' + brand + SOURCES[1] + FILE_POSIX
        extract_one_file(filepath_tb, savepath_tb, JD=False)
def reextract(text,extractor):
    dict1 = eval(text)
    opinion = dict1["opinion"]
    opinion_reextract = []
    effected_opinion = [e for e in opinion if e != []]
    for item in effected_opinion:
        for e in item:
            if e[0] == 2:
                if e[3] != "":
                    result = extractor.processing(e[3])
                    if result:
                        pass

    return opinion_reextract

def work2():
    extracter = Extractor()
    BASE_PATH = r'./labeled_review'
    files = os.listdir(BASE_PATH)
    for file in files:
        df = pd.read_csv(os.path.join(BASE_PATH,file),encoding='utf-8')
        df["re-extract"] = df.apply(lambda row: reextract(row["opinion"],extracter), axis=1)
    if not os.path.exists(r'./re-extract/'):
        os.mkdir(r'./re-extract/')
    df.to_csv(os.path.join(r'./re-extract',file),encoding='utf-8',mode='w',header=True,index=False)

def demon():
    extractor = Extractor()
    def quchong(sen, thresh=2):
        l = len(sen)
        a = [0]
        for i in range(1, l):
            if sen[i] == sen[i - 1]:
                a.append(a[i - 1] + 1)
            else:
                a.append(0)
        text = ""
        for i in range(l):
            if a[i] < thresh:
                text = text + sen[i]
        return text.lstrip()
    def truncat_text(text):
        lines = []
        for l in text:
            length = len(l)
            if length < 30:
                lines.append(l)
            else:
                i = length / 30 if length % 30 == 0 else length / 30 + 1
                for j in range(int(i)):
                    start = j * 30
                    end = (j + 1) * 30 if (j + 1) * 30 < length else length
                    lines.append(l[start:end])
        return lines
    def extract_sen(text):
        result_dicts = {}
        entities_result = []
        opinion_result = []
        print("处理前的文本", text, '\n')
        text = text.replace(' ', '')  # 去空格
        print("去空格后的文本", text, '\n')
        text = quchong(text, 2)
        print("去重后的文本", text, '\n')
        text = re.split('，|\,|。|！|\!|\.|？|\?', text)
        print("分句胡的文本", text, '\n')
        text = truncat_text(text)
        print("截断的文本", text)
        for sen in text:
            print("分析 ", sen, '\n')
            words = jieba.lcut(sen)
            for w in words:
                if w in stop:
                    words.remove(w)
            sen = ''.join(words)
            print("去停用词后的文本", sen, '\n')
            try:
                result_dict = extractor.processing(sen)
            except BaseException:
                result_dict = None
            if result_dict:
                result = parse_result(result_dict)
                entities_result.append(result['entities'])
                opinion_result.append(result['opinion'])
        result_dicts['entities'] = entities_result
        result_dicts['opinion'] = opinion_result
        return result_dicts

    def extract(text):
        try:
            result_dict = extractor.processing(text)
        except BaseException:
            result_dict = None
        if result_dict:
            result = parse_result(result_dict)
            return result
        else:
            return {}
    while True:
        print(extract_sen((input('请输入'))))







########################################################################################################
##
##          修改区结束
##
########################################################################################################

if __name__=='__main__':
    con = Controller()
    while True:
        print(con.processing(input('请输入')))
    # extractor = Extractor()
    # while True:
    #     print(extractor.processing(input('请输入')))
    # work()
    # demon()


