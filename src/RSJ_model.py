# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

from utils import *
from Helper import DataUtils


class Model(object):
    def __init__(self, config, graph):
        with graph.as_default():
            self.config = config

            self.lr = config["lr"]
            self.char_dim = config["char_dim"]
            self.lstm_dim = config["lstm_dim"]
            self.num_tags = config["num_tags"]
            self.num_chars = config['num_chars']
            self.model_type = config["model_type"]
            self.cut_num = 4
            self.cut_size = 40
            self.pos_size = 20
            self.pos_num = 123
            self.length = 70

            self.global_step = tf.Variable(0, trainable=False)
            self.best_dev_f1 = tf.Variable(0.0, trainable=False)
            self.best_test_f1 = tf.Variable(0.0, trainable=False)
            self.initializer = initializers.xavier_initializer()

            self.train_summary = ""
            self.dev_summary = ""

            # add placeholders for the model
            # 字符输入层，70个字符（batchsize, length）
            self.char_inputs = tf.placeholder(dtype=tf.int32,
                                              shape=[None, self.length],
                                              name="ChatInputs")
            # embedding层
            self.e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.length])
            # embedding层
            self.e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.length])
            # ？
            self.cut_inputs = tf.placeholder(dtype=tf.int32, shape=[None])
            # self.e1 = tf.placeholder(dtype=tf.float32,shape=[None,self.lstm_dim*2])

            # _e1 = tf.matmul(self.e1,tf.ones([self.lstm_dim*2,self.lstm_dim]))

            # self.e2 = tf.placeholder(dtype=tf.float32,shape=[None,self.lstm_dim*2])

            # _e2 = tf.matmul(self.e2, tf.ones([self.lstm_dim * 2, self.lstm_dim]))

            # ？
            self.targets = tf.placeholder(dtype=tf.int64,
                                          shape=[None],
                                          name="Targets")
            # dropout keep prob
            self.dropout = tf.placeholder(dtype=tf.float32,
                                          name="Dropout")
            # batchsize(char_input的第一个维度# )
            self.batch_size = tf.shape(self.char_inputs)[0]

            with tf.variable_scope('embeding'):
                # 一个查找表，共有1284个字符
                # tf.get_variable 获取一个已经存在的变量或者创建一个新的变量
                self.embeding = tf.get_variable('embeding', shape=[self.num_chars, self.char_dim], dtype=tf.float32,
                                                initializer=self.initializer)
                # position embedding （123,20）查找表
                pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
                pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
                # 将输入字符转为embedding
                self.char_inputs_embeding = tf.nn.embedding_lookup(self.embeding, self.char_inputs)
                # 位置embedding
                self.e1_embeding = tf.nn.embedding_lookup(pos1_embedding, self.e1)
                self.e2_embeding = tf.nn.embedding_lookup(pos2_embedding, self.e2)
                # cut embedding（4, 20）
                self.cut_embeding = tf.get_variable('cut_embedding', shape=[self.cut_num, self.cut_size],
                                                    initializer=initializers.xavier_initializer())
                # cut查找表
                cut = tf.nn.embedding_lookup(self.cut_embeding, self.cut_inputs)

            # Add model type by crownpku， bilstm or idcnn
            # parameters for idcnn
            # self.length = tf.shape(self.char_inputs)[1]
            # _e1 = tf.reshape(tf.tile(tf.reshape(_e1, [self.batch_size, self.lstm_dim]), [1, self.length,]),[self.batch_size,self.length,self.lstm_dim])
            # _e2 = tf.reshape(tf.tile(tf.reshape(_e2, [self.batch_size, self.lstm_dim]), [1, self.length,]),[self.batch_size,self.length,self.lstm_dim])
            # 网络的输入是字符的embedding，pos1，embedding，pos2 embedding， 的连接（batch, length, 140）
            inputs = tf.concat(values=[self.e1_embeding, self.char_inputs_embeding, self.e2_embeding], axis=2)
            # 经过一个双向lstm层
            if self.model_type == 'bilstm':
                # bi-directional lstm layer
                self.model_outputs = self.biRNN_layer(inputs)

            else:
                raise KeyError
            # self.logits (B,2)
            self.logits = self.project_layer_bilstm(self.model_outputs, cut)
            # 计算loss 和传入的target做onehot的那种交叉熵
            self.loss = self.loss_layer(self.logits)
            # softmax 计算出类别
            self.logits = tf.nn.softmax(self.logits)
            # 计算准确率
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits,-1),self.targets),tf.float32))

            # loss of the model
            tf.summary.scalar('accuracy', self.accuracy)
            # print(self.loss)
            tf.summary.scalar('loss', self.loss)

            # 梯度下降
            with tf.variable_scope("optimizer"):
                optimizer = self.config["optimizer"]
                if optimizer == "sgd":
                    self.opt = tf.train.GradientDescentOptimizer(self.lr)
                elif optimizer == "adam":
                    self.opt = tf.train.AdamOptimizer(self.lr)
                elif optimizer == "adgrad":
                    self.opt = tf.train.AdagradOptimizer(self.lr)
                elif optimizer == 'rmsprop':
                    self.opt = tf.train.RMSPropOptimizer(self.lr)
                else:
                    raise KeyError

                # apply grad clip to avoid gradient explosion
                grads_vars = self.opt.compute_gradients(self.loss)
                capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                     for g, v in grads_vars]
                self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

            # saver of the model
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            self.merge = tf.summary.merge_all()


    def biRNN_layer(self, model_inputs):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("BiRNN"):

            with tf.variable_scope("char_BiLSTM"):
                rnn_cell = {}
                for direction in ["forward", "backward"]:
                    with tf.variable_scope(direction):
                        # lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        #     lstm_dim,
                        #     use_peepholes=True,
                        #     initializer=self.initializer,
                        #     state_is_tuple=True)
                        rnn_cell[direction] = rnn.LSTMCell(
                            self.lstm_dim,
                            initializer=self.initializer,
                            state_is_tuple=True)
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell["forward"],
                    rnn_cell["backward"],
                    model_inputs,
                    dtype=tf.float32,)
            return tf.concat(outputs, axis=2)

    def project_layer_bilstm(self, lstm_outputs, cut, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope('attention'):
                # M:(B,L.200)->(B,L,100)
                M = tf.layers.dense(lstm_outputs,self.lstm_dim,activation=tf.nn.tanh)
                # M:(B*L，100)
                M = tf.reshape(M,[-1,self.lstm_dim])
                # W（100,1）
                W = tf.get_variable("W", shape=[self.lstm_dim, 1],
                                    dtype=tf.float32, initializer=self.initializer)
                # b（，1）
                b = tf.get_variable("b", shape=[1], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                # output : (B*L, 1)
                output = tf.nn.xw_plus_b(M,W,b)
                # length: L
                length = tf.shape(lstm_outputs)[1]
                # attention: (B, L) softmax注意力加权了，通过和一个固定的映射层
                self.attention = tf.nn.softmax(tf.reshape(output,[self.batch_size,length]))
                # c : tile扩张了 shape:(B,L,200)且200维都是attention value
                c = tf.tile(tf.reshape(self.attention, [self.batch_size, length, 1]), [1, 1, self.lstm_dim*2])
                # d:
                d = tf.reshape(c, [self.batch_size, length, self.lstm_dim*2])
                # e: (B,L,200)注意力机制
                e = d * lstm_outputs
                # attention_output 加权平均step上的向量作为输出(B, 200)
                attention_output = tf.reshape(tf.reduce_sum(e, 1),[self.batch_size,self.lstm_dim*2])
                tf.summary.histogram('attention', self.attention)

                # attention_output 连上cut向量（B,240）
                attention_output = tf.concat([attention_output,cut],-1)
            # 隐层
            with tf.variable_scope("hidden"):
                # W (240,100)
                W = tf.get_variable("W", shape=[self.lstm_dim*2+self.cut_size, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)
                # b (100)
                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                # hidden: (B, 100)
                hidden = tf.tanh(tf.nn.xw_plus_b(attention_output, W, b))


            # project to score of tags
            with tf.variable_scope("logits"):
                # W :(100,2)
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                # b(2)
                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                # pred(B, 2)
                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [self.batch_size, self.num_tags])


    def loss_layer(self, lstm_outputs, name=None):
        # vars = tf.trainable_variables()
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.01
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,logits=lstm_outputs))

    def create_feed_dict(self, type, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        chars, e1s, e2s, cut, targets = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.e1:np.asanyarray(e1s),
            self.e2:np.asanyarray(e2s),
            self.cut_inputs: np.asarray(cut),
            self.dropout: 1.0,
        }
        if type == 'train':
            feed_dict[self.targets] = np.asarray(targets)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        elif type == 'dev':
            feed_dict[self.targets] = np.asarray(targets)
        return feed_dict

    def run_step(self, sess, type, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(type, batch)
        if type == 'train':

            global_step, loss, _, summary = sess.run(
                [self.global_step, self.loss, self.train_op, self.merge],
                feed_dict)
            return global_step, loss, summary
        elif type == 'dev':
            logits, loss, accuracy, summary = sess.run([self.logits, self.loss, self.accuracy, self.merge], feed_dict)
            return logits, loss, accuracy, summary
        else:
            logits = sess.run([self.logits, self.attention], feed_dict)
            return logits


    def evaluate(self, sess, data_manager):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        losses = []
        accuracys = []
        summary = None
        for batch in data_manager.iter_batch():
            logits, loss, accuracy, summary = self.run_step(sess, 'dev', batch)
            losses.append(loss)
            results.append(logits)
            accuracys.append(accuracy)
        return np.concatenate(results,0), np.mean(losses), np.mean(accuracys),summary

    def evaluate_line(self, sess, inputs):

        return self.run_step(sess, 'test', inputs)


