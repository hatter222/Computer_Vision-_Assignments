import os
import shlex
import argparse
from tqdm import tqdm
# for python2
#import cPickle
# for python3: read in python2 pickled files
import _pickle as cPickle
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from parmap import parmap
#import scipy.spatial.distance as spdistance
from sklearn.metrics import pairwise_distances

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix',
                        default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test',
                        help='the input folder of the training images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # read labelfile
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # get filenames from labelfile
    all_files = []
    labels = []
    check = True
    for line in all_lines:
        # using shlex we also allow spaces in filenames when escaped w. ""
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # strip all known endings, note: os.path.splitext() doesnt work for
        # '.' in the filenames, so let's do it this way...
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # get now new file name
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # let's just take 100 files to speed-up the process
    max_files = 100
    indices = np.random.permutation(max_files)
    files = np.array(files)[indices]
   
    # rough number of descriptors per file that we have to load
    max_descs_per_file = int(max_descriptors / len(files))

    descriptors = []
    for i in tqdm(range(len(files))):
        with gzip.open(files[i], 'rb') as ff:
            # for python2
            # desc = cPickle.load(ff)
            # for python3
            desc = cPickle.load(ff, encoding='latin1')
            
        # get some random ones
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # TODO
    means = MiniBatchKMeans(n_clusters,

                                        compute_labels=False,

                                        batch_size=100*n_clusters).fit(descriptors).cluster_centers_
    #dummy = np.array([42])
    return  means #dummy
def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    # compute nearest neighbors
    # TODO
    K, D = clusters.shape
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = matcher.knnMatch(descriptors.astype(np.float32), 

                               clusters.astype(np.float32), k = 1)
    
    # create hard assignment
    assignment = np.zeros( (len(descriptors), len(clusters)) )
    # TODO
    for m in matches:
       if len(m) == 1:
          assignment[ m[0].queryIdx, m[0].trainIdx] = 1
    return assignment

def vlad(files, mus, powernorm=True, gmp=False, gamma=1000):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    K = mus.shape[0]
    encodings = []
    file=files[0:20]
    for f in tqdm(file):
       #print(f)
        with gzip.open(f, 'rb') as ff:
    #         print(f)
    #         # for python2:
    #         # desc = cPickle.load(ff)
    #         # for python3:
             desc = cPickle.load(ff, encoding='latin1')
             a = assignments(desc, mus)
        
             T,D = desc.shape
             f_enc = np.zeros( (K,D), dtype=np.float32)
             diff=np.zeros( (1,D), dtype=np.float32)
    
             for k in range(mus.shape[0]):
        #     # it's faster to select only those descriptors that have
        #     # this cluster as nearest neighbor and then compute the 
        #     # difference to the cluster center than computing the differences
        #     # first and then select
       #idx=np.array(a[:,k]).nonzero()
       #if idx 
                f_enc[k] =a[:,k] .T.dot(desc)-mus[k] 
       #diff = (np.sum(desc[idx,:] - mus[k,:], axis=0)).reshape(1,D)
       #f_enc[k,:]=diff#np.random(0,5 [1,64],dtype=np.float32)#np.ravel(diff)
        # c) power normalization
        
                enc=[f_enc[k]]
                enc = np.concatenate(enc)
                if powernorm:
       #  #     # TODO
                   enc = np.sign(enc)*np.sqrt(np.abs(enc))     
                else:      
        # # l2 normalization
        # # TODO 
          #encodings = preprocessing.normalize(encodings, norm='l2', copy=False) 
                    enc = enc/np.sqrt(np.dot(enc,enc))
                    f_enc[k]=enc
        encodings.append(f_enc.reshape(1,-1))
    encodings=np.vstack(encodings)    
    print(encodings.shape)
    return encodings # encodings

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    num_pos=1#encs_test[0].shape[0]
    num_neg=encs_train.shape[0]
    # set up labels
    # TODO
    y = np.hstack((1,-np.ones(num_neg)))
    
    def loop(i):
        # compute SVM 
        # and make feature transformation
        X = np.vstack((encs_test[i], encs_train))
        classifier = LinearSVC(C=C, dual=False,verbose=1,class_weight='balanced')
        classifier.fit(X,y)
        #x = classifier.decision_function(X)
        
        x = normalize(classifier.coef_, norm='l2') 
        # TODO
        return x

    # let's do that in parallel: 
    # if that doesn't work for you, just exchange 'parmap' with 'map'
    # Even better: use DASK arrays instead, then everything should be
    # parallelized
    new_encs = list(map( loop, tqdm(range(len(encs_test)))))
    new_encs = np.concatenate(new_encs, axis=0)
    # return new encodings
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        desc = cPickle.load(ff) #, encoding='latin1')
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # compute cosine distance = 1 - dot product between l2-normalized
    # descriptors
    # TODO
    #desc = cPickle.load(ff, encoding='latin1')
    
    #for i in range(len(encs)):
    dists=1-np.dot(encs,encs.T)
#pairwise_distances(encs[i],encs[i],metric="cosine")
       
    
    # mask out distance with itself
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # sort each row of the distance matrix
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions)
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    print('Top-1 accuracy: {} - mAP: {}'.format(float(correct) / n_encs, mAP))


if __name__ == '__main__':
    #parser = argparse.ArgumentParser('retrieval')
    #parser = parseArgs(parser)
    #args = parser.parse_args()
    np.random.seed(42) # fix random seed
   
    # a) dictionary
    files_train, labels_train = getFiles('icdar17_local_features/train/', '_SIFT_patch_pr.pkl.gz',
                                         'icdar17_local_features/icdar17_labels_train.txt')
    print('#train: {}'.format(len(files_train)))
    if not os.path.exists('mus.pkl.gz'):
        # TODO
        descriptors=loadRandomDescriptors(files_train, 500000)
        print('> loaded {} descriptors:'.format(len(descriptors)))
        
        # cluster centers
        mus=dictionary(descriptors, 100)
        
        print('> compute dictionary')
        # TODO
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)

  
    # b) VLAD encoding
    print('> compute VLAD for test')
    files_test, labels_test = getFiles('icdar17_local_features/test/', '_SIFT_patch_pr.pkl.gz','icdar17_local_features/icdar17_labels_test.txt')
    print('#test: {}'.format(len(files_test)))
    fname = 'enc_test_gmp{}.pkl.gz'.format(1) if 'store_true' else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or 'store_true':
    # #     # TODO      
       enc_test=vlad(files_test,mus,powernorm=True,gmp=False, gamma=1000)
       with gzip.open(fname, 'wb') as fOut:
              cPickle.dump(enc_test, fOut, -1)
    else:
           with gzip.open(fname, 'rb') as f:
               enc_test = cPickle.load(f)
   
     # cross-evaluate test encodings

    print('> evaluate')
    evaluate(enc_test, labels_test)

    # d) compute exemplar svms
    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(1) if 'store_true' else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or 'store_true':
         # TODO
       enc_train=vlad(files_train,mus,powernorm=True,gmp=False, gamma=1000)
       with gzip.open(fname, 'wb') as fOut:
             cPickle.dump(enc_train, fOut, -1)
    else:
      with gzip.open(fname, 'rb') as f:
           enc_train = cPickle.load(f)
    print('> esvm computation')
    # # TODO
    enc_test=esvm(enc_test, enc_train, C=1000)
    
    # # eval
    evaluate(enc_test, labels_test)
    print('> evaluate')
