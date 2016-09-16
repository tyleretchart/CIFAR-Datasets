import numpy as np
import time
import os.path
import urllib
import tarfile
import matplotlib.pyplot as plt

class cifar10:
    
    def __init__(self):    
        if not os.path.exists("cifar-10-batches-py"):
            print "Downloading dataset from 'https://www.cs.toronto.edu/~kriz'..."
            urllib.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
            print "Unzipping tar file..."
            tar = tarfile.open('cifar-10-python.tar.gz', "r:gz")
            tar.extractall()
            tar.close()
            print "Deleting tar file..."
            os.remove("cifar-10-python.tar.gz")
            print "Done"
            
        self.train = _training()
        self.test = _testing()
        
        data = _unpickle('cifar-10-batches-py/batches.meta')
        self.key = data['label_names']
        
    def printPicture(self, subject):
        subjectR = subject[0:1024].reshape(32, 32)
        subjectG = subject[1024:2048].reshape(32, 32)
        subjectB = subject[2048:3072].reshape(32, 32)
        plotSubject = np.array([subjectR.T, subjectG.T, subjectB.T]).T
        plt.imshow(plotSubject)
        plt.tight_layout()
    
    
class _training:

    def __init__(self):
        data = _unpickle('cifar-10-batches-py/data_batch_1')
        self.features = data['data']
        self.labels = data['labels']
        self.labels = np.atleast_2d( self.labels ).T

        data = _unpickle('cifar-10-batches-py/data_batch_2')
        self.features = np.append(self.features, data['data'], axis=0)
        data_labels = np.atleast_2d( data['labels'] ).T
        self.labels = np.append(self.labels, data_labels, axis=0)
       
        data = _unpickle('cifar-10-batches-py/data_batch_3')
        self.features = np.append(self.features, data['data'], axis=0)
        data_labels = np.atleast_2d( data['labels'] ).T
        self.labels = np.append(self.labels, data_labels, axis=0)    
        
        data = _unpickle('cifar-10-batches-py/data_batch_4')
        self.features = np.append(self.features, data['data'], axis=0)
        data_labels = np.atleast_2d( data['labels'] ).T
        self.labels = np.append(self.labels, data_labels, axis=0)
        
        data = _unpickle('cifar-10-batches-py/data_batch_5')
        self.features = np.append(self.features, data['data'], axis=0)
        data_labels = np.atleast_2d( data['labels'] ).T
        self.labels = np.append(self.labels, data_labels, axis=0)
        
        self.shuffle()
        self.batch_index = 0
             
    def shuffle(self):
        np.random.seed(int(time.time()))
        index = np.arange(len(self.labels))
        index = np.random.permutation(index)
        self.features = self.features[index]
        self.labels = self.labels[index]
        
    def next_batch(self, batch_size):
        if (batch_size + self.batch_index > len(self.features)): self.batch_index = 0
        batchFeatures = self.features[self.batch_index : self.batch_index+batch_size]
        batchLabels = self.labels[self.batch_index : self.batch_index+batch_size]
        self.batch_index += batch_size
        return batchFeatures, batchLabels
       
    
class _testing:

    def __init__(self):
        data = _unpickle('cifar-10-batches-py/test_batch')
        self.features = data['data']
        self.labels = data['labels']
        self.labels = np.atleast_2d( self.labels ).T
        

def _unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
