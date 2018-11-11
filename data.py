import tensorflow as tf
import pickle
import numpy as np

# Class for creatinng dataset
class ImageNetDataset(object):
    def __init__(self, cfg, mode, path):
        self.cfg = cfg
        self.mode = mode
        self.path = path

        # Unpickle the dataset from the given path
        self.dict = self.unpickle(self.path)

        # Create a batch dataset object from the dictonary with images and one hot labels
        self.dataset = tf.data.Dataset.from_tensor_slices((self.dict[b'data'] , self.dict[b'labels'] ))
        self.dataset = self.dataset.map(lambda row, label: ( tf.reshape(row, [3,32,32] ) , label) )
        self.dataset = self.dataset.map(lambda row, label: ( tf.transpose(row ,perm=[1,2,0] ) , label) )
        self.dataset = self.dataset.map(lambda row, label: ( tf.image.resize_images(row, [224,224]) ,tf.one_hot(label,self.cfg.NUM_CLASSES)) )
        self.dataset = self.dataset.map(lambda image, one_hot: ( tf.subtract(tf.to_float(image),self.cfg.DATA_MEAN),   one_hot) )
        self.dataset = self.dataset.shuffle(1024).batch(self.cfg.BATCH_SIZE)

    # Unpickle function for extracting the dataset

    def unpickle(self,path):

        # If it is training combine the dataset from 1 to 4
        if self.mode == 'train':
            with open(path+'1', 'rb') as fo:
                d=pickle.load(fo, encoding='bytes')
            for i in range(2,5):
                p = path+str(i)
                with open(p, 'rb') as fo:
                    d=self.update(d, pickle.load(fo, encoding='bytes'))
            return d

        # If validation then take the data from given path
        elif self.mode == 'val':
            with open(path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        elif self.mode == 'test':
            with open(path, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

    # Function to concatenate the data from different training files
    
    def update(self, d1,d2):
        d1[b'labels'] = np.concatenate( (d1[b'labels'], d2[b'labels']) )
        d1[b'data'] = np.concatenate( (d1[b'data'], d2[b'data']) )
        return d1