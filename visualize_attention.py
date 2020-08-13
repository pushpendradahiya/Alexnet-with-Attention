import re
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from data import ImageNetDataset
from config import Configuration
from models.alexnet import AlexNet
import cv2


tfe.enable_eager_execution()

class Tester(object):

    def __init__(self, cfg, net, testset):

        self.cfg = cfg
        self.net = net
        self.testset = testset

        # Load the model back

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.cfg.LEARNING_RATE, momentum=self.cfg.MOMENTUM)
        self.checkpoint_dir = self.cfg.CKPT_PATH
        self.checkpoint_encoder = os.path.join(self.checkpoint_dir, 'Model')
        self.root1 = tfe.Checkpoint(optimizer=self.optimizer, model=self.net, optimizer_step=tf.train.get_or_create_global_step())
        self.root1.restore(tf.train.latest_checkpoint(self.checkpoint_dir))


    def test(self, mode):

    	# get the image, label and attention maps for layer 4 and 5 
        for (ex_i, (image, label)) in enumerate(tfe.Iterator(self.testset.dataset),1):
            if ex_i==5:
                out ,Att1, Att2 = self.net(image)
                image = image[0]
                break

        # Reshape for attention on layer 5
        A = tf.reshape(Att2, [6,6]).numpy()

        # Reshape for attention on layer 4
        B = tf.reshape(Att1, [13,13]).numpy()

        # Normalize the map obtained
        out = np.zeros(A.shape, np.double)
        A=cv2.normalize(A, out, 0.0, 1.0, cv2.NORM_MINMAX)

        # Add mean back to the data image
        image=image+self.cfg.DATA_MEAN
        image=image.numpy()
        out1 = np.zeros(image.shape, np.double)
        image=cv2.normalize(image, out1, 0.0, 1.0, cv2.NORM_MINMAX)

        # Resize attentin map
        A=cv2.resize(A,(image.shape[1],image.shape[0]))

        # Get image and map in 0 to 255 range
        A = np.uint8(255 * A)
        image = np.uint8(255 * image)

        # Get color map of the image
        A = cv2.applyColorMap(A, cv2.COLORMAP_JET)

        # Weighted addition of the image and map
        superimposed_img = cv2.addWeighted(image, 0.6, A, 0.4, 0)

        cv2.imshow('map',A)
        cv2.imshow("GradCam", superimposed_img)
        cv2.imshow('image',image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

	# Call tester on test data

    cfg = Configuration()
    net = AlexNet(cfg, training=False)
    path = 'cifar-10-batches-py/test_batch'
    testset = ImageNetDataset(cfg, 'test', path)


    tester = Tester(cfg, net, testset)

    tester.test('test')
