import os
import json
import shutil
import logging
import numpy as np
import pandas

import tensorflow as tf
from conlleval import return_report

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# def test_ner(results, path):
#     """
#     Run perl script to evaluate model
#     """
#     script_file = "conlleval"
#     output_file = os.path.join(path, "ner_predict.utf8")
#     result_file = os.path.join(path, "ner_result.utf8")
#     with open(output_file, "w") as f:
#         to_write = []
#         for block in results:
#             for line in block:
#                 to_write.append(line + "\n")
#             to_write.append("\n")
#
#         f.writelines(to_write)
#     os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
#     eval_lines = []
#     with open(result_file) as f:
#         for line in f:
#             eval_lines.append(line.strip())
#     return eval_lines


def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = path+"_predict.utf8"
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines



def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir("../maps"):
        os.makedirs("../maps")
    if not os.path.isdir("../log"):
        os.makedirs("../log")
    if not os.path.isdir("../config"):
        os.makedirs("../config")
    if not os.path.isdir("../ckpt"):
        os.makedirs("../ckpt")


def clean(params):
    """
    Clean current folder
    remove saved model and training log 
    """
    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path[:-6]):
        shutil.rmtree(params.ckpt_path[:-6])

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir(params.log_file):
        shutil.rmtree(params.log_file)

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)





def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def extract0(sentence,tags):
    entities = []
    entity = ""
    chunk_start = False
    chunk_end = False
    for i in range(len(tags)):
        if tags[i][0] == "S":
            entities.append({"value": sentence[i], "start": i, "end": i+1, "type":tags[i][2:]})
            continue
        if i==0 and tags[i][0]!='O': chunk_start = True
        if tags[i][0] == 'B':chunk_start = True
        if i>0:
            if tags[i-1] == 'O' and tags[i][0] == 'I': chunk_start = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'S': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'I': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'S': chunk_end = True

        if chunk_end or chunk_start:

            if chunk_end:
                entities[-1]['value'] = entity
                entities[-1]['end'] = i
                chunk_end = False
                entity = ""
            if chunk_start:
                entities.append({'type': tags[i][2:], 'start': i})
                entity = sentence[i]
                chunk_start = False

        elif entity:
            entity+=sentence[i]

        if entity and i+1==len(tags):
            # entity+=sentence[i]
            entities[-1]['value'] = entity
            entities[-1]['end'] = i+1
            chunk_end = False
            entity = ""
    return entities

def extract(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"value": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"value": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item['entities']

def rel_extract(entities,tags,distance,same_type=False):
    """
    extract related type entity from entities,tag's type must be in tags and the distance between entities less than 'distance'   
    :param entities: 
    :param tags: 
    :param distance: 
    :return: 
    """
    assert isinstance(entities,list),'entities must be a list'
    assert isinstance(tags,list),'tags must be a list'
    # 实体序列的二维关系矩阵
    length = len(entities)
    rels = np.zeros((length,length),np.int32)
    # print(entities)
    for i in range(length):
        if entities[i]['type'] in tags:
            for j in range(i+1,length):
                if entities[j]['type'] in tags and (entities[j]['start']-entities[i]['end'])<distance \
                        and entities[i]['value']!=entities[j]['value']:
                    if same_type or entities[i]['type']!=entities[j]['type']:
                        rels[i,j] = 1

    return rels

def item_prf(logits,targets,id_tag):
    shape = np.asarray(logits).shape
    logits = np.argmax(logits, 1)
    onehot_l = np.eye(shape[0], shape[1])[logits]
    onehot_t = np.eye(shape[0], shape[1])[targets]
    classify_c = np.sum(onehot_l * onehot_t, 0)
    classify_t = np.sum(onehot_t, 0)
    classify_l = np.sum(onehot_l, 0)
    item_p = classify_c / classify_l
    item_r = classify_c / classify_t
    item_f = 2*item_p*item_r / (item_p+item_r)
    item_pd = {}
    item_rd = {}
    item_fd = {}
    for i,v in enumerate(item_f):
        item_pd[id_tag[i]] = item_p[i]
        item_rd[id_tag[i]] = item_r[i]
        item_fd[id_tag[i]] = v
    return item_pd,item_rd,item_fd

def Confusion_matrix(logits,targets,tags):
    df = pandas.DataFrame(data=np.zeros(shape=[len(tags),len(tags)]),index=tags,columns=tags)
    for i in range(logits.shape[0]):
        df.ix[targets[i],logits[i]]+=1
    return df


if __name__=='__main__':
#     v = rel_extract([{'end': 4, 'start': 2, 'value': '充电', 'type': 'pair-feature'}, {'end': 7, 'start': 4, 'value': '特别快', 'type': 'pair-feature'}, {'end': 10, 'start': 8, 'value': '颜值', 'type': 'pair-feature'}, {'end': 12, 'start': 10, 'value': '很高', 'type': 'pair-perspective'}]
# ,['pair-feature','pair-perspective'],3,True)
#     print(v)
#     print(np.where(v>0))
    a = [1,0,2]
    b = [1,1,2]
    tags = ['a','b','c']
    print(Confusion_matrix(a,b,tags))