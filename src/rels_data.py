from utils import *
import numpy as np
import json
with open('/home/shaohui/Documents/毕设/project/data/RSE/mobile.RSE.test',encoding='utf8') as r,open('/home/shaohui/Documents/毕设/project/'
                                                                                                    'data/RSJ/mobile.RSJ.test','w',encoding='utf8') as w:
    for chars in r.read().split('\n\n'):
        if chars=='':
            continue
        sentence = []
        tags = []
        for word in chars.rstrip().split('\n'):
            ws = word.split(' ')
            # print(ws)
            if len(ws)>2:
                sentence.append(' ')
                tags.append(ws[-1])
            else:
                # print(ws)
                char,tag = ws
                sentence.append(char)
                tags.append(tag)
        entitys = extract(sentence,tags)
        rel_matrix = rel_extract(entitys,['pair-perspective','pair-feature'],10,False)
        xs,ys = np.where(rel_matrix>0)
        print(xs,ys)
        if len(xs)==0:
            continue
        rels = []
        for x,y in zip(xs,ys):
            rel = {}
            rel['start'] = str(x)
            rel['end'] = str(y)
            rel['value'] = entitys[x]['value']+'_'+entitys[y]['value']
            rel['rel_type'] = "positive"
            rels.append(rel)
        print(rels)
        w.write(''.join(sentence)+'\n')
        w.write(json.dumps(entitys,ensure_ascii=False,sort_keys=True)+'\n')
        w.write(json.dumps(rels,ensure_ascii=False,sort_keys=True)+'\n\n')
