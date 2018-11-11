import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Class for Alexnet model

class AlexNet(tf.keras.Model):

	def __init__(self, cfg, training):
		super(AlexNet, self).__init__()

		self.cfg = cfg
		self.training = training

		# Convolutional layers 1 to 5

		conv_init = tf.contrib.layers.xavier_initializer_conv2d()

		self.conv1 = tf.layers.Conv2D(96, 11, 4, 'SAME',activation=tf.nn.relu,kernel_initializer=conv_init)
		self.pool1 = tf.layers.MaxPooling2D(3, 2, 'VALID')

		self.conv2 = tf.layers.Conv2D(256, 5, 1, 'SAME',activation=tf.nn.relu,kernel_initializer=conv_init)
		self.pool2 = tf.layers.MaxPooling2D(3, 2, 'VALID')

		self.conv3 = tf.layers.Conv2D(384, 3, 1, 'SAME',activation=tf.nn.relu,kernel_initializer=conv_init)

		self.conv4 = tf.layers.Conv2D(384, 3, 1, 'SAME',activation=tf.nn.relu,kernel_initializer=conv_init)

		self.conv5 = tf.layers.Conv2D(256, 3, 1, 'SAME',activation=tf.nn.relu,kernel_initializer=conv_init)
		self.pool5 = tf.layers.MaxPooling2D(3, 2, 'VALID')

		# Fully connected layers

		fc_init = tf.contrib.layers.xavier_initializer()

		self.fc1 = tf.layers.Dense(4096,activation=tf.nn.relu,kernel_initializer=fc_init)
		self.drop1 = tf.layers.Dropout(self.cfg.DROPOUT)

		self.fc2 = tf.layers.Dense(4096,activation=tf.nn.relu,kernel_initializer=fc_init)
		self.drop2 = tf.layers.Dropout(self.cfg.DROPOUT)

		self.out = tf.layers.Dense(self.cfg.NUM_CLASSES,kernel_initializer=fc_init)


	def call(self, x):
		# Function that executes the model on call

		output = self.conv1(x)
		output = tf.nn.lrn(output, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
		output = self.pool1(output)

		output = self.conv2(output)
		output = tf.nn.lrn(output, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
		output = self.pool2(output)

		output = self.conv3(output)

		output = self.conv4(output)

		output = self.conv5(output)
		output = self.pool5(output)

		output = tf.layers.flatten(output)

		output = self.fc1(output)

		# Execute dropout if training
		if self.training:
			output = self.drop1(output)

		output = self.fc2(output)
		if self.training:
			output = self.drop2(output)

		output = self.out(output)

		return output