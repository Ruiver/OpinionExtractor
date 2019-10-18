# encoding=utf8
import os
import pickle
from collections import OrderedDict
import json
from RSJ_Helper import *
from RSJ_model import Model
from utils import *
import tensorflow as tf
import numpy as np
from Arguments import Arguments


class RSJ(object):
    def __init__(self,args):
        config  = {
            "num_tags":2,
            "train_file":"/home/shaohui/Documents/毕设_lsh/project/ZH/data/RSJ/mobile.RSJ.train.new1",
            "dev_file":"/home/shaohui/Documents/毕设_lsh/project/ZH/data/RSJ/mobile.RSJ.dev.new1",
            "test_file":"/home/shaohui/Documents/毕设_lsh/project/ZH/data/RSJ/mobile.RSJ.test.new1",
        }
        self.graph = tf.Graph()
        self.args = args
        self.args.dicts(config)
        make_path(self.args)
        self.logger = get_logger(self.args.log_file)
        self.create_helper()


    def create_helper(self,helper=None):
        if helper:
            self.helper = helper
            return
        self.helper = RSJ_Helper()


        # make path for store log and model if not exist
        if os.path.isfile(self.args.config_file):
            self.args.load_config(self.args.config_file)
        else:
            self.args.config_model(self.helper.ner.char_to_id, self.helper.tag_id)

            self.args.save_config()
        self.args.print_config(self.logger)

        self.model = None

    def save_model(self):
        self.model.saver.save(self.sess, self.args.ckpt_path)
        self.logger.info("model saved")

    def create_model(self):
        # create model, reuse parameters if exists
        self.model = Model(self.args.config, graph = self.graph)
        ckpt = tf.train.get_checkpoint_state(self.args.ckpt_path[:-6])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.model.embeding.assign(self.helper.embeding))

    def train(self):
        # self.create_helper()
        self.helper.load_train(self.args.train_file)
        self.helper.load_dev(self.args.dev_file)
        self.helper.load_test(self.args.test_file)

        # self.helper.select_tag_schema(self.args.tag_schema)
        self.helper.create_batch(self.args.batch_size)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = self.helper.train_batch.len_data
        with tf.Session(config=tf_config,graph=self.graph) as self.sess:
            self.create_model()
            self.logger.info("start training")
            loss = []
            self.train_summary = tf.summary.FileWriter(self.args.summary_path + '/train', self.graph)
            self.dev_summary = tf.summary.FileWriter(self.args.summary_path + '/dev', self.graph)
            self.test_summary = tf.summary.FileWriter(self.args.summary_path + '/test', self.graph)
            for i in range(self.args.max_epoch):
                for batch in self.helper.train_batch.iter_batch(shuffle=True):
                    self.step, batch_loss, summary = self.model.run_step(self.sess, 'train', batch)

                    loss.append(batch_loss)
                    if self.step % self.args.steps_check == 0:
                        iteration = self.step // steps_per_epoch + 1
                        self.train_summary.add_summary(summary, self.step)
                        self.logger.info("iteration:{} step:{}/{}, "
                                    "{} loss:{:>9.6f}".format(
                            iteration,  self.step % steps_per_epoch, steps_per_epoch, self.args.task_type, np.mean(loss)))
                        loss = []

                best = self.evaluate("dev")
                if best:
                    self.save_model()
                self.evaluate("test")

    def evaluate(self, name):
        self.logger.info("evaluate:{}".format(name))
        ner_results,losses,accuracy,summary = self.model.evaluate(self.sess, self.helper.__dict__[name+'_batch'])
        f1 = float(accuracy)
        self.logger.info("{} loss:{:>9.6f} accuracy:{}".format(
         self.args.task_type, np.mean(losses),accuracy))
        if name == "dev":
            # summary = tf.Summary(value=[
            #     tf.Summary.Value(tag="dev_loss", simple_value=np.mean(losses)),
            # ])
            self.dev_summary.add_summary(summary, self.step)
            best_test_f1 = self.model.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_dev_f1, f1).eval()
                self.logger.info("new best dev accuracy score:{:>.3f}".format(f1))
            return f1 > best_test_f1
        elif name == "test":
            # summary = tf.Summary(value=[
            #     tf.Summary.Value(tag="test_loss", simple_value=np.mean(losses)),
            # ])
            self.test_summary.add_summary(summary, self.step)
            best_test_f1 = self.model.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_test_f1, f1).eval()
                self.logger.info("new best test accuracy score:{:>.3f}".format(f1))
            return f1 > best_test_f1

    def restore_model(self):
        self.args.load_config(self.args.config_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # print(char_to_id,id_to_char,tag_to_id)
        self.sess = tf.Session(config=tf_config,graph=self.graph)
        self.create_model()
    def evaluate_line(self,sentence):
        # 文本进入一个ner识别模型返回ner标注的结果
        result = self.helper.input_from_line(sentence)
        """返回{"label":True,'entities':entities, 'rels':rels,'rsj_data':BatchManager.pad_data(data)}
            其中label表示这句话中NER模型识别出了实体，并判断有潜在匹配的实体对，rels就是潜在匹配的实体对，
            rsj_data就是包含了一组实体的相关特征的特征数据，用于RSJ分类模型判定是否匹配
        """
        # print(result)
        if not result['label']:
            return {'sentence':sentence,'entities':result['entities'],'rels':[]}
        """logits由RSJ实体匹配模型判断是否两个实体匹配，返回logits 和attention"""
        logits,attention = self.model.evaluate_line(self.sess, result['rsj_data'])
        result.pop('rsj_data')
        # logits_id = np.argmax(logits,-1)
        for i,logit in enumerate(logits):
            # a_sorted = sorted(enumerate(list(attention[i])),key=lambda a:a[1],reverse=True)
            # print([[sentence[a_sorted[j][0]],a_sorted[j][1]] for j in range(3)])
            # print([(w,s)for w,s in zip(list('1'+sentence+'2'),attention[i])])
            # result['rels'][i]['rel_type_pro'] = {t:logit[int(j)] for j,t in self.helper.id_tag.items()}
            """logit第一列表示不匹配的logits分数，第二列是匹配的logits
               分数,按照这个原则排，logit_sorted就是一个列表，每一项是（i,logits）的形式，排序后，排在第一位的i变成0或1 
            """
            logit_sorted = sorted(list(enumerate(logit)),key=lambda v:v[1],reverse=True)
            # print(logit_sorted)
            # result['rels'][i]['rel_type'] = [self.helper.id_tag[logit_sorted[i][0]] for i in range(1)]
            """得到是否匹配的结果"""
            result['rels'][i]['rel_type'] = self.helper.id_tag[logit_sorted[0][0]]
            # if self.helper.id_tag[logit_sorted[0][0]] != 'none':
            #     print(result['rels'][i])


        return result



if __name__=="__main__":
    args = Arguments('RSJ')
    clean(args)
    rsj = RSJ(args)
    rsj.train()
    rsj.restore_model()
    rsj.helper.load_test(args.test_file)
    test_batch = BatchManager(rsj.helper.test_data, len(rsj.helper.test_data))
    logits,_,accuracy,_ = rsj.model.evaluate(rsj.sess, test_batch)
    targets = [batch for batch in test_batch.iter_batch()]
    print(np.argmax(logits,axis=1),targets[0][3])
    print(Confusion_matrix(np.argmax(logits,axis=1),targets[0][4],list(rsj.helper.id_tag.values())))

    print(item_prf(logits,targets[0][3],rsj.helper.id_tag))

    while True:
        print(json.dumps(rsj.evaluate_line(input("请输入")),ensure_ascii=False))