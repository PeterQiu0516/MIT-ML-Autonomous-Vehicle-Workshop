from get_f import get_features
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn import svm
import random
import pandas as pd
import ImageFunctions as imagefunctions

# Load pickled data
data_all = pickle.load(open('data.p', 'rb'))

X_all, y_all = data_all['images'], data_all['labels']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))
print 'Original training samples: ' + str(len(X_train))
# convert to numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# data augmentation
X_rot = []
y_rot = []
for X,y in zip(X_train,y_train):
    for r in range(1,4):
        imrot = np.rot90(X,r)
        X_rot.append(imrot)
        y_rot.append(y)
X_train = np.append(X_train, X_rot, axis=0)
y_train = np.append(y_train, y_rot)

# check data
# Number of training examples# Numbe
n_train = X_train.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# Shape of traffic sign image
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
classes = np.unique(y_train)

print('Images loaded.')
print "Training samples: " + str(n_train)
print "Testing samples: " + str(n_test)
print "Image data shape: " + str(image_shape)
print "Classes: " + str(classes) + "\n"
# ------------------------------------------------------------------ #

# Pre-Process
## Pre-Process: RGB
X_train_prep = imagefunctions.preprocess_rgb(X_train)
X_test_prep = imagefunctions.preprocess_rgb(X_test)
## Pre-Process: Grayscale
#X_train_prep = imagefunctions.preprocess_grayscale(X_train)
#X_test_prep = imagefunctions.preprocess_grayscale(X_test)

# check quality after pre-processing
check_quality = False
if (check_quality):
    index = random.randint(0, len(X_train))
    print("Random Test for {0}".format(y_train[index]))
    plt.figure(figsize=(5,5))

    plt.subplot(1, 2, 1)
    plt.imshow(X_train[index].squeeze())
    plt.title("Before")

    plt.subplot(1, 2, 2)
    if (proc_num_channels==1):
        plt.imshow(X_train_prep[index].squeeze(), cmap="gray")
    else:
        plt.imshow(X_train_prep[index].squeeze())
    plt.title("After")
    plt.show()

# replace data with preprocessed images
X_train = X_train_prep
X_test = X_test_prep

print('Pre-processing done.')

# ------------------------------------------------------------------ #

def getFeatures(img):
    return get_features(img)
    ######################################################################################

    # TO-DO: Feature engineering - assemble a "feature vector" for SVM to maximize accuracy
    # Note: features must be numerical values, i.e .feature vector is a vector of numbers.
    # Below is a 2-dimensional example:
    #####################################################################################
    out = img[::2,::2,:].ravel()

    return out
    
    return [
             imagefunctions.num_red_pixels(img),
             imagefunctions.num_white_pixels(img),
	     imagefunctions.num_edges(img),
	     imagefunctions.num_corners(img),
    ]

# Extract features
Features_train = []
features = []
for x,y in zip(X_train,y_train):
    features = getFeatures(x)  #[num_corners(x),num_red_pixels(x)]#,num_white_pixels(x)]
    Features_train.append(features)

def normalize_features(feature_vector,fmn,fsd):
    numDim = len(feature_vector)
    normFeatures = []
    normfeat = [None]*numDim
    for i in range(numDim):
        normfeat[i] = (feature_vector[i]-fmn[i])/fsd[i]
    normFeatures.append(normfeat)
    #transpose result
    res = np.array(normFeatures).T
    return res

# normalize features
from operator import itemgetter
numDim = len(Features_train[0])
scalefeat = [None]*numDim
fmean = np.mean(np.asarray(Features_train),axis=0)
fstd = np.std(np.asarray(Features_train),axis=0)

# assemble data for plotting
f0 = [] # no-sign
f1 = [] # stop sign
f2 = [] # warning sign
fvec=[]
for ff,y in zip(Features_train,y_train):
    if y==0:
        ffnorm = normalize_features(ff,fmean,fstd)
        fvec.append(ffnorm)
        f0.append(ffnorm)
    elif y==1:
        ffnorm = normalize_features(ff,fmean,fstd)
        fvec.append(ffnorm)
        f1.append(ffnorm)
    else:
        ffnorm = normalize_features(ff,fmean,fstd)
        fvec.append(ffnorm)
        f2.append(ffnorm)

normFeatures = np.squeeze(fvec)
print('Feature extraction done.')
# ------------------------------------------------------------------ #

clf = svm.SVC() #(kernel='rbf')
clf.fit(normFeatures,y_train)

svmdata = {
    #"clf": clf,
    "fmean" : fmean,
    "fstd" : fstd
#    "scalefeat" : scalefeat
}
pickle.dump(clf, open('model_svm.p', 'wb'))
pickle.dump(svmdata, open('svm_params.p', 'wb'))
# with open('model_svm.p', 'wb') as handle:
#     pickle.dump(svmdata, handle, protocol=2)
#pickle.dump(X_test, open('Xtest.p', 'wb'))
#pickle.dump(y_test, open('ytest.p', 'wb'))
print('SVM training done.')

#######################################################
# visualize (first two dimensions/features only)
visualize = False
if visualize:
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(f0)[:,0], np.asarray(f0)[:,1], color='g', marker='o', label='no sign')
    ax.scatter(np.asarray(f1)[:,0], np.asarray(f1)[:,1], color='r', marker='*', label='stop')
    ax.scatter(np.asarray(f2)[:,0], np.asarray(f2)[:,1], color='b', marker='^', label='warn')
    # ax.scatter(np.asarray(plotf0)[:,0], np.asarray(plotf0)[:,1], color='g', marker='o', label='no sign')
    # ax.scatter(np.asarray(plotf1)[:,0], np.asarray(plotf1)[:,1], color='r', marker='*', label='stop')
    # ax.scatter(np.asarray(plotf2)[:,0], np.asarray(plotf2)[:,1], color='b', marker='^', label='warn')
    ax.legend()
    ax.grid(True)
    plt.show()

#######################################################

# test data

del clf
del fmean
del fstd
print 'SVM model and params deleted'

clf = pickle.load(open('model_svm.p', 'rb'))
svmparams = pickle.load(open('svm_params.p', 'rb')) #pickle.load(f2)
fmean = svmparams['fmean']
fstd = svmparams['fstd']
print 'SVM model reloaded'

del X_test
del y_test

# Load pickled data
data_test = pickle.load(open('test.p', 'rb'))
X_test, y_test = data_test['images'], data_test['labels']

# convert to numpy
X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = imagefunctions.preprocess_rgb(X_test)

# test
fvec=[]
for x in X_test:
    feat = getFeatures(x)
    normfeat = normalize_features(feat,fmean,fstd)
    testvec = np.asarray(normfeat).reshape(-1,1)
    fvec.append(testvec)
rvec = clf.predict(np.array(fvec).squeeze())
res = [a==b for (a,b) in zip(y_test,rvec)]
print 'Test results:'
print 'right = ' + str(sum(res)) + ' wrong = ' + str(len(y_test)-sum(res))
print 'accuracy = ' + str(100.*sum(res)/(len(y_test))) + ' %'
