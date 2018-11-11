import re
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from data import ImageNetDataset
from config import Configuration
from models.alexnet import AlexNet

tfe.enable_eager_execution()

# Class for Tester
class Tester(object):

	def __init__(self, cfg, net, testset):

		self.cfg = cfg
		self.net = net
		self.testset = testset

		# Restore the model 
		self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.cfg.LEARNING_RATE, momentum=self.cfg.MOMENTUM)
		self.checkpoint_dir = self.cfg.CKPT_PATH
		self.checkpoint_encoder = os.path.join(self.checkpoint_dir, 'Model')
		self.root1 = tfe.Checkpoint(optimizer=self.optimizer, model=self.net, optimizer_step=tf.train.get_or_create_global_step())
		self.root1.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

	# Function for top-1  Accuracy and error
	def top_1_accuracy(self, x, y):
		pred = tf.nn.softmax(self.net(x))

		top_1_accuracy_value = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.int64),
							tf.argmax(y, axis=1, output_type=tf.int64)),dtype=tf.float32))

		return top_1_accuracy_value

	# Function for test
	def test(self, mode):
		test_examples = 10000

		total_top1_accuracy = 0.

		# Iterate over the dataset
		for (ex_i, (images, label)) in enumerate(tfe.Iterator(self.testset.dataset),1):
			# Call the top-1 helper function
			top_1_a = self.top_1_accuracy(images, label)
			total_top1_accuracy += top_1_a

		# Print the final accuracy

		print ('Top-1: {:.4f} '.format(total_top1_accuracy / test_examples))
		print ('Top-1 error rate: {:.4f} '.format(1 - (total_top1_accuracy / test_examples)))

if __name__ == '__main__':
	i = 0

	# Path for test results
	if not os.path.exists("Tests"):
		os.makedirs('Tests')

	while os.path.exists("Tests/Test%s.txt" % i):
		i += 1

	LOG_PATH = "Tests/Test%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	cfg = Configuration()

	# Get the Alexnet form models
	net = AlexNet(cfg, training=False)

	# Path for test dataset
	path = 'cifar-10-batches-py/test_batch'
	testset = ImageNetDataset(cfg, 'test', path)

	# Create a tester object
	tester = Tester(cfg, net, testset)

	# Call test function on tester object
	tester.test('test')