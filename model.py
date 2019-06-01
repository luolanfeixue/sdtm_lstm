# coding=utf-8
import numpy as np
import os
import tensorflow as tf
import time
import losses as ranking_losses


class Model():
	
	def __init__(self, learning_rate, lstm_size, num_layers, train_keep_prob=0.5, grad_clip=5, batch_size=32,
	             time_steps=182, input_size=26, output_size=13):
		self.learning_rate = learning_rate
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.train_keep_prob = train_keep_prob
		self.grad_clip = grad_clip
		self.batch_size = batch_size
		self.time_steps = time_steps
		self.input_size = input_size
		self.output_size = output_size
		self.saver = None
		self.sess = None
		self.debug_sess = None
	
	def add_placeholders(self):
		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.time_steps, self.input_size],
			                             name='inputs')
			self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_size], name='label')
			# self.inputs = tf.placeholder(tf.float32, shape=[None, None, None], name='inputs')
			# self.labels = tf.placeholder(tf.float32, shape=[None, None], name='label')
			self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
			self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
	
	def build_model(self):
		"""
		建立模型的计算图
		:return:
		"""
		self.add_placeholders()
		
		# 定义lstmcell
		def get_a_call(lstm_size, keep_prob):
			lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, initializer=tf.orthogonal_initializer())
			drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
			return drop
		
		with tf.name_scope('lstm'):
			# 组织多层lstm
			cell = tf.nn.rnn_cell.MultiRNNCell(
				[get_a_call(self.lstm_size, self.train_keep_prob) for _ in range(self.num_layers)])
			# 将lstm在时间维度上展开
			self.initial_state = cell.zero_state(self.batch_size, tf.float32)
			# lstm_output shape (self.batch_size, self.time_steps, self.lstm_size)
			lstm_output, _ = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
			lstm_output = lstm_output[:self.batch_size, -1, :self.lstm_size]
			# lstm_output shape (self.batch_size, self.lstm_size)
			self.lstm_output = tf.reshape(lstm_output, [-1, self.lstm_size])
		
		with tf.variable_scope('dense'):
			dense_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.output_size], stddev=0.1))
			dense_b = tf.Variable(tf.zeros(self.output_size))
		
		self.logits = tf.matmul(self.lstm_output, dense_w) + dense_b
		self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')
		
		with tf.name_scope('loss'):
			loss_fn = ranking_losses.make_loss_fn(ranking_losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS)
			self.loss = loss_fn(self.labels, self.logits)
		
		self.add_train_op('adam', self.learning_rate, self.loss, self.grad_clip)
	
	def add_train_op(self, lr_method, lr, loss, clip):
		_lr_m = lr_method.lower()
		with tf.variable_scope('train_step'):
			if _lr_m == 'adam':
				optimizer = tf.train.AdamOptimizer(lr)
			elif _lr_m == 'adagrad':
				optimizer = tf.train.AdagradDAOptimizer(lr)
			elif _lr_m == 'sgd':
				optimizer = tf.train.GradientDescentOptimizer(lr)
			elif _lr_m == 'rmsprob':
				optimizer = tf.train.RMSPropOptimizer(lr)
			else:
				raise NotImplementedError('Unknown methond {}'.format(_lr_m))
			
			if clip > 0:
				grads, vs = zip(*optimizer.compute_gradients(loss))
				grads, gnorm = tf.clip_by_global_norm(grads, clip)
				self.train_op = optimizer.apply_gradients(zip(grads, vs))
			else:
				self.train_op = optimizer.minimize(loss)
	
	def initialize_session(self):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
	
	def train(self, training_generater, dev_generater, epoch, save_best_path, save_path, save_every_n, log_every_n):
		"""
		训练模型
		:param batch_generator: 数据迭代器
		:param epoch: 轮数
		:param save_path: 模型保存地址
		:param save_every_n: 每n轮保存一次
		:param log_every_n:每n轮打一次log
		:return:
		"""
		self.initialize_session()
		# self.restore_session(save_best_path)
		step = 0
		new_state = self.sess.run(self.initial_state)
		sum_mydef_loss = 0
		index = 0
		for x, y in training_generater:
			index += len(x)
			step += 1
			start = time.time()
			feed = {self.inputs: x, self.labels: y,
			        self.keep_prob: self.train_keep_prob,
			        self.lr: self.learning_rate,
			        self.initial_state: new_state}
			_, batch_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
			mydef_loss = self.predict_batch(x, y)
			sum_mydef_loss += mydef_loss
			end = time.time()
			if step % log_every_n == 0:
				print('step: {}/{}... '.format(step, epoch),
				      'loss: {:.4f}... '.format(batch_loss),
				      'mydef_loss: {:.4f}... '.format(sum_mydef_loss / index),
				      '{:.4f} sec/batch'.format((end - start)))
			if (step % save_every_n == 0):
				self.save_session(save_path, step)
			if step >= epoch:
				break
		eval = self.evaluate(dev_generater)
		print('dev loss', eval)
		self.save_session(save_path, step)
		self.close_session()
	
	def save_session(self, save_path, step):
		"""Saves session """
		# if not os.path.exists(save_path):
		# 	os.makedirs(save_path)
		self.saver.save(self.sess, save_path, global_step=step)
	
	def close_session(self):
		"""Closes the session"""
		self.sess.close()
	
	def restore_session(self, save_path):
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		save_path = tf.train.latest_checkpoint(save_path)
		self.saver.restore(self.sess, save_path)
	
	def predict_batch(self, x):
		pass
	
	def evaluate_batch(self, x, y):
		sum_loss = 0
		feed = {self.inputs: x, self.keep_prob: 1.0}
		# shape (batch_size,list_size)
		proba_prediction = self.sess.run([self.proba_prediction], feed_dict=feed)
		proba_prediction = np.array(proba_prediction).reshape(self.batch_size, self.output_size)
		# print(proba_prediction)
		# print(y)
		pred_topks = self.sess.run(tf.nn.top_k(proba_prediction, k=5).indices).tolist()
		label_topks = self.sess.run(tf.nn.top_k(y.astype(np.int32), k=5).indices).tolist()
		proba_label_zipped = zip(pred_topks, label_topks)
		for pred_topk, label_topk in proba_label_zipped:
			# print('pred_topk',pred_topk)
			# print('label_topk',label_topk)
			loss = self.calculate_loss(pred_topk, label_topk)
			print(loss)
			sum_loss += loss
		return sum_loss
	
	def evaluate(self, batch_generator):
		sum_loss = 0
		index = 0
		for x, y in batch_generator:
			index += len(x)
			loss = self.evaluate_batch(x, y)
			sum_loss += loss
		return sum_loss / index
	
	def calculate_loss(self, pred_topk, label_topk):
		weight_sum = 0
		for i, pred in enumerate(pred_topk):
			weight = 1.0 / (len(pred_topk))
			label_set = set(label_topk[0:i + 1])
			weight_score = 1
			# if pred in label_set or (pred + 1) in label_set or (pred - 1) in label_set:
			if pred in label_set:
				weight_score = 0
			weight = weight_score * weight
			weight_sum += weight
		return weight_sum
