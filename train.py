import os.path
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from models.alexnet import AlexNet
from data import ImageNetDataset
from config import Configuration
import utils as ut


# Enablling Eager execution

tfe.enable_eager_execution()

# Train class for training the model
class Trainer(object):

	def __init__(self, cfg, net, trainingset, valset, resume=False):
		self.cfg = cfg
		self.net = net

		# Datasets
		self.trainingset = trainingset
		self.valset = valset

		# Using Adam optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.LEARNING_RATE)
		
		# Create global step
		self.global_step = tf.train.get_or_create_global_step()

		# Create checkpoint directory and save checkpoints
		self.epoch = tfe.Variable(0, name='epoch', dtype=tf.float32, trainable=False)
		self.checkpoint_dir = self.cfg.CKPT_PATH
		self.checkpoint_encoder = os.path.join(self.checkpoint_dir, 'model')
		self.root1 = tfe.Checkpoint(optimizer=self.optimizer, model=self.net, optimizer_step=tf.train.get_or_create_global_step())
		
		# If resume is true continue from saved checkpoint
		if resume:
			self.root1.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

	# Loss calculaions
	def loss(self, mode, x, y):

		# Get predicted labels
		pred = self.net(x)

		# Get the losses using cross entropy loss
		loss_value = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=pred)

		return loss_value

	# Accuracy Calulations
	def accuracy(self, mode, x, y):

		# Get predicted labels
		pred = tf.nn.softmax(self.net(x))

		# Calculate accuracy using average of correct predictions over all data points
		accuracy_value = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.int64),
							tf.argmax(y, axis=1, output_type=tf.int64)),dtype=tf.float32)) / float(pred.shape[0].value)

		return accuracy_value

	# Training function contining training loop
	def train(self):
		# Get start time
		start_time = time.time()
		step_time = 0.0

		# Run training loop for the number of epochs in configuration file
		for e in range(int(self.epoch.numpy()), self.cfg.EPOCHS):
			tf.assign(self.epoch, e)

			# Run the iterator over the training dataset
			for (batch_i, (images, labels)) in enumerate(tfe.Iterator(self.trainingset.dataset)):
				self.global_step = tf.train.get_global_step()
				step = self.global_step.numpy() + 1

				step_start_time = int(round(time.time() * 1000))

				# Define optimizer
				self.optimizer.minimize(lambda: self.loss('train', images, labels), global_step=self.global_step)


				step_end_time = int(round(time.time() * 1000))
				step_time += step_end_time - step_start_time

				# If it is display step find training accuracy and print it
				if (step % self.cfg.DISPLAY_STEP) == 0:
					l = self.loss('train', images, labels)
					a = self.accuracy('train', images, labels).numpy()
					print ('Epoch: {:03d} Step/Batch: {:09d} Step mean time: {:04d}ms \n\tLoss: {:.7f} Training accuracy: {:.4f}'.format(e, int(step), int(step_time / step), l, a))

				# If it is Validation step find validation accuracy on valdataset and print it
				if (step % self.cfg.VALIDATION_STEP) == 0:
					val_images, val_labels = tfe.Iterator(self.valset.dataset).next()
					l = self.loss('val', val_images, val_labels)
					a = self.accuracy('val', val_images, val_labels).numpy()
					int_time = time.time() - start_time
					print ('Elapsed time: {} --- Loss: {:.7f} Validation accuracy: {:.4f}'.format(ut.format_time(int_time), l, a))

				# If it is save step, save checkpoints
				if (step % self.cfg.SAVE_STEP) == 0:
					encoder_path = self.root1.save(self.checkpoint_encoder)				
		# Save the varaibles at the end of training step
		encoder_path = self.root1.save(self.checkpoint_encoder)				
		print('\nVariables saved\n')


if __name__ == '__main__':
	i = 0
	# Make dir for logs
	if not os.path.exists("logs"):
		os.makedirs('logs')

	while os.path.exists("logs/log%s.txt" % i):
		i += 1

	# Initialize log path
	LOG_PATH = "logs/log%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	# Get the configuration
	cfg = Configuration()
	net = AlexNet(cfg, training=True)

	# If it is resume task, make it true
	resume  = False

	# Path for train dataset
	path = 'cifar-10-batches-py/data_batch_'

	# Get the data set using data.py
	trainingset = ImageNetDataset(cfg, 'train',path)

	# Path for valdidation dataset
	path_val = 'cifar-10-batches-py/data_batch_1'
	valset = ImageNetDataset(cfg, 'val', path_val)

	# Make the Checkpoint path
	if not os.path.exists(cfg.CKPT_PATH):
		os.makedirs(cfg.CKPT_PATH)

	# Make an object of class Trainer
	trainer = Trainer(cfg, net, trainingset, valset, resume)

	# Call train function on trainer class
	trainer.train()