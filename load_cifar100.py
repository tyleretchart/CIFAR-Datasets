import numpy as np
import time
import os.path
import urllib
import tarfile
import matplotlib.pyplot as plt

class cifar100:
    
    def __init__(self):    
        if not os.path.exists("cifar-100-python"):
            print "Downloading dataset..."
            urllib.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", "cifar-100-python.tar.gz")
            print "Unzipping tar file..."
            tar = tarfile.open('cifar-100-python.tar.gz', "r:gz")
            tar.extractall()
            tar.close()
            print "Deleting tar file..."
            os.remove("cifar-100-python.tar.gz")
            print "Done"
            
        self.train = _training()
        self.test = _testing()
        
        data = _unpickle('cifar-100-python/meta')
        self.fineKey = data['fine_label_names']
        self.coarseKey = data['coarse_label_names']
        
    def printPicture(self, subject):
        subjectR = subject[0:1024].reshape(32, 32)
        subjectG = subject[1024:2048].reshape(32, 32)
        subjectB = subject[2048:3072].reshape(32, 32)
        plotSubject = np.array([subjectR.T, subjectG.T, subjectB.T]).T
        plt.imshow(plotSubject)
        plt.tight_layout()
    
    
class _training:

    def __init__(self):
        data = _unpickle('cifar-100-python/train')
        self.features = data['data']
        self.fineLabels = data['fine_labels']
        self.fineLabels = np.atleast_2d(self.fineLabels).T
        self.coarseLabels = data['coarse_labels']
        self.coarseLabels = np.atleast_2d(self.coarseLabels).T
        self.shuffle()
        self.batch_index = 0
             
    def shuffle(self):
        np.random.seed(int(time.time()))
        index = np.arange(len(self.fineLabels))
        index = np.random.permutation(index)
        self.features = self.features[index]
        self.fineLabels = self.fineLabels[index]
        self.coarseLabels = self.coarseLabels[index]
        
    def next_batch(self, batch_size):
        if (batch_size + self.batch_index > len(self.features)): self.batch_index = 0
        batchFeatures = self.features[self.batch_index : self.batch_index+batch_size]
        batchFineLabels = self.fineLabels[self.batch_index : self.batch_index+batch_size]
        batchCoarseLabels = self.coarseLabels[self.batch_index : self.batch_index+batch_size]
        self.batch_index += batch_size
        return batchFeatures, batchFineLabels, batchCoarseLabels
       
    
class _testing:

    def __init__(self):
        data = _unpickle('cifar-100-python/test')
        self.features = data['data']
        self.fineLabels = data['fine_labels']
        self.fineLabels = np.atleast_2d(self.fineLabels).T
        self.coarseLabels = data['coarse_labels']
        self.coarseLabels = np.atleast_2d(self.coarseLabels).T
        

def _unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict