# coding=utf-8
import os
import datetime
import pickle
import numpy as np
import copy


def batch_generator(arr, batch_size, time_steps):
	np.random.seed(87)
	np.random.shuffle(arr)
	for n in range(0, arr.shape[0], batch_size):
		x = arr[n: n + batch_size, 182 - time_steps:-1].astype(np.int32)
		y = arr[n: n + batch_size, -1, 8:20].astype(np.int32)
		if x.shape[0] == batch_size and y.shape[0] == batch_size:
			yield x, y


class DataCore(object):
	
	def __init__(self, filename_root=None, filename_origin=None, filename_output=None):
		# 如果之前已经有过数据处理，则直接加载
		# filename = filename_root + filename_output
		if filename_output is not None:
			filename = filename_root + filename_output
			if os.path.exists(filename):
				with np.load(filename) as data:
					self.arr_np = data['data']
		
		else:
			# 数据处好后变成numpy格式，并压缩存储到文件中
			self.arr_np = self.convert2np(filename_root, filename_origin, filename_output)
		print('数据准备完成, arr_np.shape', self.arr_np.shape)
	
	def get_data(self):
		return self.arr_np
	
	def train_test_split(self, train_rate, seed):
		"""
		将数据分割为train和test
		:param train_rate:
		:return:
		"""
		np.random.seed(seed)
		indices = np.random.permutation(self.arr_np.shape[0])
		training_count = (int)(self.arr_np.shape[0] * train_rate)
		training_idx, dev_idx = indices[:training_count], indices[training_count:]
		training, dev = self.arr_np[training_idx, :], self.arr_np[dev_idx, :]
		return training, dev
	
	def convert2np(self, filename_root=None, filename_origin=None, filename_output=None):
		"""
		将一个pin到所有日期数据补全，如果一个pin到某天没有点击，则将该天数据补0.
		:param filename_root: 项目跟目录
		:param filename_origin: 原始数据文件
		:param filename_output: 输出数据文件
		:return:
		"""
		
		filename_origin = filename_root + filename_origin
		pin2value = self.data_2_dict(filename_origin)
		print('pin数据量为：', len(pin2value))
		pin2data_final = dict()
		for pin, dt2data in pin2value.items():
			data_sortedby_dt = []
			for date in self.get_date_list('20180601', '20181130'):
				if date in dt2data:
					data_sortedby_dt.append(dt2data[date].split('|'))
				else:
					week = datetime.datetime.strptime(date, '%Y%m%d').weekday()
					strvalue = '0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|' + str(week) + '|0'
					data_sortedby_dt.append(strvalue.split('|'))
			if '20181130' in dt2data:
				flag = False
				values = data_sortedby_dt[-1][8:20]
				for value in values:
					value = (int)(value)
					if value > 0:
						flag = True
				if flag is True:
					pin2data_final[pin] = data_sortedby_dt
		print('对pin补充缺失数据完成,最终训练pin数据量为', len(pin2data_final))
		values = []
		for pin, value in pin2data_final.items():
			values.append(value)
		arr_np = np.array(values)
		if filename_output is not None:
			filename_output = filename_root + filename_output
			np.savez_compressed(filename_output, data=arr_np)
		print('将数据压缩存储')
		return arr_np
	
	def data_2_dict(self, filename):
		"""
		将一个pin的数据整合到一起。key为pin，value就是这个pin所有日期到点击数据。
		:param filename: 原始数据文件，pin,当天点击情况数据,dt。
		当天点击情况数据：24小时的点击情况，今天是周几，该天是否有数据。一工26个格子。
		:return:
		"""
		pin2value = dict()
		with open(filename, 'r') as f:
			for line in f:
				line = line.strip().split('\t')
				pin = line[0]
				data = line[1]
				dt = line[2]
				if pin in pin2value:
					dt2data = pin2value.get(pin)
				else:
					dt2data = {}
				dt2data[dt] = data
				pin2value[pin] = dt2data
		return pin2value
	
	def get_date_list(self, datestart=None, dateend=None):
		"""
		:param datestart: 开始日期
		:param dateend: 结束日期
		:return: 从开始日期到结束日期的日期list
		"""
		# 创建日期辅助表
		if datestart is None:
			datestart = '20170101'
		if dateend is None:
			dateend = datetime.datetime.now().strftime('%Y%m%d')
		
		# 转为日期格式
		datestart = datetime.datetime.strptime(datestart, '%Y%m%d')
		dateend = datetime.datetime.strptime(dateend, '%Y%m%d')
		date_list = []
		date_list.append(datestart.strftime('%Y%m%d'))
		while datestart < dateend:
			# 日期叠加一天
			datestart += datetime.timedelta(days=+1)
			# 日期转字符串存入列表
			date_list.append(datestart.strftime('%Y%m%d'))
		return date_list
