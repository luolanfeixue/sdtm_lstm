import tensorflow as tf
from data_core import DataCore, batch_generator
from model import Model
import os

FLAGS = tf.flags.FLAGS

# 数据文件参数
# 数据文件参数
tf.flags.DEFINE_string('filename_origin', 'data/sdtm_final.txt', '原始文件')
tf.flags.DEFINE_string('filename_root', '/home/hhl/code/sdtm_lstm/', '项目跟目录')
# tf.flags.DEFINE_string('filename_output', 'data/sdtm_output.npz', '输出np文件')

# 模型参数
tf.flags.DEFINE_integer('time_steps', 180, '时间序列长度')
tf.flags.DEFINE_integer('input_size', 26, '输入序列的长度')
tf.flags.DEFINE_integer('output_size', 12, '输出序列的长度')
tf.flags.DEFINE_integer('lstm_size', 4, '输入序列的长度')
tf.flags.DEFINE_integer('num_layers', 4, '层数')

tf.flags.DEFINE_integer('batch_size', 64, '一个batch')
tf.flags.DEFINE_float('learning_rate', 0.01, '学习率')
tf.flags.DEFINE_float('train_keep_prob', 1.0, 'dropout rate during training')
tf.flags.DEFINE_integer('grad_clip', 3, '梯度clip值')



def main(_):
	data_core = DataCore(
		filename_root=FLAGS.filename_root,
		filename_origin=FLAGS.filename_origin)
	data = data_core.get_data()
	batch_size = data.shape[0]
	
	model = Model(
		learning_rate=FLAGS.learning_rate,
		lstm_size=FLAGS.lstm_size,
		num_layers=FLAGS.num_layers,
		train_keep_prob=FLAGS.train_keep_prob,
		grad_clip=FLAGS.grad_clip,
		batch_size=batch_size,
		time_steps=FLAGS.time_steps,
		input_size=FLAGS.input_size,
		output_size=FLAGS.output_size)
	model.build_model()
	model.restore_session(FLAGS.save_path)
	loss = model.evaluate(dev_generator)
	print(loss)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	tf.app.run()