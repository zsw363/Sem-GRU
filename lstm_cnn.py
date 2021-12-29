# -*- coding utf-8 -*-
"""
Create on 2020/12/19 14:49
@author: zsw
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training,
                 cnn_input = 0, wubi_input = 0, is_cnn = False, is_gru = False):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.emb_size = 768
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.is_cnn = is_cnn
        self.is_gru = is_gru
        self.input_picture = cnn_input
        self.wubi_gru = wubi_input
        self.weight = 0

    def create_cnn(self, X, b_alpha=0.1, keep_prob=0.75, image_height=50, image_width=50, emb_size=768):
        #input: X:[v]*batch_size*height*width*seq_length
        #return:[batch_size, lengths*emb_size]

        #create_model
        x = tf.reshape(X, shape=[-1, image_height, image_width, self.seq_length])
        x = tf.to_float(x)
        # print(">>> input x: {}".format(x))


        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, self.seq_length, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 8], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(b_alpha * tf.random_normal([8]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

#         # 卷积层3
#         wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
#                               initializer=tf.contrib.layers.xavier_initializer())
#         bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
#         conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
#         conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#         conv3 = tf.nn.dropout(conv3, keep_prob)
#         # print(">>> convolution 3: ", conv3.shape)
#         next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

#         # 全连接层1
#         wd1 = tf.get_variable(name='wd1', shape=[next_shape, 128], dtype=tf.float32,
#                                   initializer=tf.contrib.layers.xavier_initializer())
#         bd1 = tf.Variable(b_alpha * tf.random_normal([128]))
#         dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
#         dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
#         dense = tf.nn.dropout(dense, keep_prob)
#         # 全连接层2
#         wout = tf.get_variable('name', shape=[128, self.seq_length * emb_size], dtype=tf.float32,
#                                    initializer=tf.contrib.layers.xavier_initializer())
#         bout = tf.Variable(b_alpha * tf.random_normal([self.seq_length * emb_size]))

#         with tf.name_scope('y_prediction'):
#             y_predict = tf.add(tf.matmul(dense, wout), bout)

        # 全连接层1
        next_shape = conv2.shape[1] * conv2.shape[2] * conv2.shape[3]
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, self.seq_length * emb_size], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(b_alpha * tf.random_normal([self.seq_length * emb_size]))
        dense = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, keep_prob)
        y_predict = dense

        return y_predict

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # project
            logits = self.project_layer(self.embedded_chars)
            # blstm
            lstm_output = self.blstm_layer(logits)
            
#             next_shape = lstm_output.shape[1]*lstm_output.shape[2]
#             lstm_output = tf.reshape(lstm_output, [-1, next_shape])

#             # 调整形状
            b_alpha = 0.1
#             keep_prob = 0.75
#             # 全连接层
#             wd = tf.get_variable(name='wd_lstm', shape=[next_shape, self.seq_length * self.num_labels], dtype=tf.float32,
#                                  initializer=tf.contrib.layers.xavier_initializer())
#             bd = tf.Variable(b_alpha * tf.random_normal([self.seq_length * self.num_labels]))
#             dense = tf.nn.relu(tf.add(tf.matmul(lstm_output, wd), bd))
#             logit = tf.nn.dropout(dense, keep_prob)

#             logits = tf.reshape(logit, shape=[-1, self.seq_length, self.num_labels])

            
            
            # 全连接层
            wd = tf.get_variable(name='wd_lstm', shape=[self.hidden_unit*2, self.num_labels], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            bd = tf.Variable(b_alpha * tf.random_normal([self.num_labels]))
               
            logits = tf.nn.xw_plus_b(lstm_output,wd,bd)
            # crf
            loss, trans, _ = self.crf_layer(logits)
            # CRF decode, pred_ids 是一条最大概率的标注路径
            pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
            return (loss, logits, trans, pred_ids, wd)


    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw


    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers == 3:
                cell_fw = [rnn.DropoutWrapper(rnn.GRUCell(size), output_keep_prob=self.dropout_rate) for size in [self.hidden_unit,356,356]]
                cell_bw = [rnn.DropoutWrapper(rnn.GRUCell(size), output_keep_prob=self.dropout_rate) for size in [self.hidden_unit,356,356]]
                cell_fw = rnn.MultiRNNCell(cell_fw)
                cell_bw = rnn.MultiRNNCell(cell_bw)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_layer(self, embedded_chars, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        cov_layer_num = 1
        bert_outputs = tf.expand_dims(embedded_chars, -1)
        logits = bert_outputs
        if self.is_cnn:
            y_predict = self.create_cnn(self.input_picture, emb_size=self.emb_size)
            cnn_output = tf.reshape(y_predict, [-1, self.seq_length, self.emb_size])
            cnn_output = tf.expand_dims(cnn_output, -1)
            logits = tf.concat(axis=-1, values=[cnn_output, logits])
            cov_layer_num += 1
        if self.is_gru:
            x = tf.reshape(self.wubi_gru, shape=[-1, self.seq_length, 4])
            x = tf.to_float(x)

            logit_gru = rnn.GRUCell(self.emb_size)
            logit_gru = rnn.DropoutWrapper(logit_gru, output_keep_prob=self.dropout_rate)
            logit_gru, _ = tf.nn.dynamic_rnn(logit_gru, inputs=x, dtype=tf.float32)

            gru_output = tf.reshape(logit_gru, [-1, self.seq_length, self.emb_size])
            gru_output = tf.expand_dims(gru_output, -1)
            logits = tf.concat(axis=-1, values=[gru_output, logits])
            cov_layer_num += 1


        def cnn_layer(conv, cov_layer_num, keep_prob=0.75, b_alpha=0.1):
            
            wc_1 = tf.get_variable(name='cov_fea_1', shape=[3, 3, cov_layer_num, 8], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            bc_1 = tf.Variable(b_alpha * tf.random_normal([8]))
            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv, wc_1, strides=[1, 1, 1, 1], padding='SAME'), bc_1))
            conv = tf.nn.dropout(conv, keep_prob)

            wc_2 = tf.get_variable(name='cov_fea_2', shape=[3, 3, 8, 1], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())
            bc_2 = tf.Variable(b_alpha * tf.random_normal([1]))
            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv, wc_2, strides=[1, 1, 1, 1], padding='SAME'), bc_2))
            conv = tf.nn.dropout(conv, keep_prob)
            
#             if 0:
#                 for i in range(n_layer-1):
#                     wc = tf.get_variable(name='cov_fea_'+str(i+2), shape=[3, 3, 1, 1], dtype=tf.float32,
#                                          initializer=tf.contrib.layers.xavier_initializer())
#                     bc = tf.Variable(b_alpha * tf.random_normal([1]))
#                     conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv, wc, strides=[1, 1, 1, 1], padding='SAME'), bc))
#                     conv = tf.nn.dropout(conv, keep_prob)
            return conv
        #卷积聚焦特征
        x = tf.reshape(logits, shape=[-1, self.seq_length, self.emb_size, cov_layer_num])
        x = tf.to_float(x)
        b_alpha = 0.1
        keep_prob = 0.75


        conv = cnn_layer(x, cov_layer_num, b_alpha=b_alpha, keep_prob=keep_prob)

        logit = conv
#         logit = x
        
#         next_shape = conv.shape[1]*conv.shape[2]*conv.shape[3]
#         conv = tf.reshape(conv, (-1, next_shape))
    
#         #调整形状
#         # 全连接层
#         wd_1 = tf.get_variable(name='wd_1', shape=[next_shape, 1024], dtype=tf.float32,
#                              initializer=tf.contrib.layers.xavier_initializer())
#         bd_1 = tf.Variable(b_alpha * tf.random_normal([1024]))
#         dense = tf.nn.relu(tf.add(tf.matmul(conv, wd_1), bd_1))
#         conv = tf.nn.dropout(dense, keep_prob)
#         # 全连接层2
#         wd_2 = tf.get_variable(name='wd_2', shape=[1024, self.seq_length*self.emb_size], dtype=tf.float32,
#                               initializer=tf.contrib.layers.xavier_initializer())
#         bd_2 = tf.Variable(b_alpha * tf.random_normal([self.seq_length*self.emb_size]))
#         dense = tf.nn.relu(tf.add(tf.matmul(conv, wd_2), bd_2))
#         logit = tf.nn.dropout(dense, keep_prob)
        
        
        
        
        
#         logit = x
#         # 全连接层
#         wd_1 = tf.get_variable(name='wd_1', shape=[3, 1], dtype=tf.float32,
#                              initializer=tf.contrib.layers.xavier_initializer())
#         bd_1 = tf.Variable(b_alpha * tf.random_normal([1]))
#         dense = tf.nn.relu(tf.add(tf.matmul(logit, wd_1), bd_1))
#         logit = tf.nn.dropout(dense, keep_prob)
        

        return tf.reshape(logit, [-1, self.seq_length, self.emb_size])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans, None
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans, -log_likelihood


