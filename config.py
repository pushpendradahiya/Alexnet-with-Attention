# Configuration used for training

class Configuration(object):
	DATA_PATH = 'cifar-10-batches-py'
	# CIFAR-10 data mean
	DATA_MEAN = [125.3, 123.0, 113.9]
	IMG_SHAPE = [224, 224, 3]
	NUM_CLASSES = 10

	# Training hyperparameters
	LEARNING_RATE = 1e-4
	MOMENTUM = 0.9
	BATCH_SIZE = 128
	EPOCHS = 50

	# Display steps
	DISPLAY_STEP = 100
	VALIDATION_STEP = 500
	SAVE_STEP = 5000

	# Paths for checkpoint
	CKPT_PATH = 'ckpt'
	SUMMARY_PATH = 'summary'

	# Net architecture hyperparamaters
	LAMBDA = 5e-4 #for weight decay
	DROPOUT = 0.5

	# Test hyperparameters
	K_PATCHES = 5
	TOP_K = 1