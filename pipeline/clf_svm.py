import numpy as np
import cv2
from sklearn import svm
from scipy.ndimage.measurements import label
import ImageFunctions as imagefunctions
from tracker import Tracker
import pickle
import random
from get_f import get_features

###########################################################3
# sliding window parameters
imgsize = (128,96)
windowsize = (50,50)
slidestep = (5,5) # number of pixels to slide window
min_red_pixels = 20 # min red pixel to process window

class SVMCLF(object):
    def __init__(self, fn_model, fn_params, nhistory=1):

        # Trained SVM model
        fn_model = 'model_svm.p'
        fn_params = 'svm_params.p'
        self.clf = pickle.load(open(fn_model, 'rb'))
        svmparams = pickle.load(open(fn_params, 'rb')) #pickle.load(f2)
        self.fmean = svmparams['fmean']
        self.fstd = svmparams['fstd']

        # Settings
        self.nhistory = 1 # tracker buffer
        self.dt = 0.1 # update interval

        ############################################################################
        # TO-DO: adjust thresholds
        ############################################################################
        self.K_detthresh_stop = 0.1 #0.05 # detection threshold (factor of window count)
        self.K_detthresh_warn = 2.5 # detection threshold (factor of window count)
        self.K_mapthresh = 0.08 # discard threshold (factor of window count)
        #self.K_stopbias = 0.4 # bias to favor STOP over WARN (factor of window count)

        # tracker
        self.tracker = Tracker(self.nhistory)

    def getFeatures(self,img):
        ############################################################################
        # TO-DO: use the same feature vector that was used for training
        ############################################################################
	return get_features(img)
	out = img[:,:2,:].ravel()
	return out
        return [
            imagefunctions.num_red_pixels(img),
            imagefunctions.num_white_pixels(img),
	    imagefunctions.num_edges(img),
	    imagefunctions.num_corners(img)
        ]

    def normalize_features(self,feature_vector,fmn,fsd):
        numDim = len(feature_vector)
        normFeatures = []
        normfeat = [None]*numDim
        for i in range(numDim):
            normfeat[i] = (feature_vector[i]-fmn[i])/fsd[i]
        normFeatures.append(normfeat)
        #transpose result
        res = np.array(normFeatures).T
        return res

    def draw_labeled_bboxes(self,img, labels, boxcolor):
        # Iterate through all detected cars
        for item_number in range(1, labels[1]+1):
            # Find pixels with each item_number label value
            nonzero = (labels[0] == item_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], boxcolor, 2)
        # Return the image
        return img

    def search_windows(self,img, windows,framenum = 0):
        # preprocess frame
        img_prep = imagefunctions.preprocess_one_rgb(img[0:127][:])
        fvec=[]
        for window in windows:
            # extract test window from image
            test_img = img_prep[window[0][1]:window[1][1], window[0][0]:window[1][0]]
            # extract features
            feat = self.getFeatures(test_img)
            # normalize features
            normfeat = self.normalize_features(feat,self.fmean,self.fstd)
            # assemble batch
            testvec = np.asarray(normfeat).reshape(1,-1)
            fvec.append(testvec)

        # batch prediction
        if (np.array(fvec).ndim==3):
            rvec = self.clf.predict(np.array(fvec).squeeze(axis=1))
        else:
            rvec = []

        # list of positive stop sign detection windows
        stop_indices = [i for i, x in enumerate(rvec) if x==1]
        stop_windows = [windows[i] for i in stop_indices]

        # list of positive warn sign detection windows
        warn_indices = [i for i, x in enumerate(rvec) if x==2]
        warn_windows = [windows[i] for i in warn_indices]

        # return positve detection windows
        return stop_windows, warn_windows

    def find_signs(self,img):
        startx = 0 #60
        stopx = imgsize[0]-windowsize[0] #80
        starty = 0 #20 #19
        stopy =imgsize[1]-windowsize[1] #30

        window_list = []
        for x in range(startx, stopx, slidestep[0]):
            for y in range(starty, stopy, slidestep[1]):
                img_in = img[ y:y+windowsize[1], x:x+windowsize[0]]
                #img_crop_pp = imagefunctions.preprocess_one_rgb(img_crop)
                #img_in = np.array(255*img_crop_pp, dtype=np.uint8)
                if (imagefunctions.num_red_pixels(img_in)>min_red_pixels):
                    window_list.append(((x, y), (x+windowsize[0], y+windowsize[1])))

        #stop_windows, warn_windows = self.search_windows(img, window_list, framenum=random.randint(0,9999))
        stop_windows, warn_windows = self.search_windows(img, window_list)

        # if no window to search
        numwin = len(window_list)
        if (numwin == 0):
            decision = 0
            labels = [None]
            return decision, labels, img

        # Method 1 - Count windows
#        if ((len(stop_windows)<2) and (len(warn_windows)<2)):
#            return 0,[None]
#        elif (len(stop_windows)>=len(warn_windows)):
#            return 1,[None]
#        else:
#            return 2,[None]

        # Method 2 - Localized heatmap based decision
        heat_stop = np.zeros_like(img[:,:,0]).astype(np.float)
        heat_warn = np.zeros_like(img[:,:,0]).astype(np.float)
        for bbox in window_list:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            #cv2.rectangle(img,(startx, starty),(endx, endy),(200,0,0),1)
        for bbox in warn_windows:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            heat_warn[starty:endy, startx:endx] += 1.
            cv2.rectangle(img,(startx, starty),(endx, endy),(0,255,0),1)
        for bbox in stop_windows:
            startx = bbox[0][0]
            starty = bbox[0][1]
            endx = bbox[1][0]
            endy = bbox[1][1]
            heat_stop[starty:endy, startx:endx] += 1.
            cv2.rectangle(img,(startx, starty),(endx, endy),(255,0,0),1)

        score_stop = np.max(heat_stop)
        score_warn = np.max(heat_warn)

        # ---- GET DECISION ---- #
        decision = self.get_decision(score_stop, score_warn, numwin)

        # plot final decision region
        mapthresh = self.K_mapthresh * numwin
        labels=[None]
        if (decision == 1):
            heatmap_stop = heat_stop
            heatmap_stop[heatmap_stop <= mapthresh] = 0
            labels = label(heatmap_stop)
        elif (decision == 2):
            heatmap_warn = heat_warn
            heatmap_warn[heatmap_warn <= mapthresh] = 0
            labels = label(heatmap_warn)

        return decision, labels, img

    def get_decision(self, score_stop, score_warn, numwin):
        # decision thresholds and biases
        detthresh_stop = self.K_detthresh_stop * numwin
        detthresh_warn = self.K_detthresh_warn * numwin

        ############################################################################
        # TO-DO: design a decision maker
        ############################################################################
        # Make Decision
        if (score_stop>=detthresh_stop): # and (score_stop + stopbias > score_warn):
            decision = 1
        elif (score_warn>=detthresh_warn):
            decision = 2
        else:
            decision = 0

        return decision

    def processOneFrame(self,img):
        # get decision for current frame
        dec, labels, draw_img = self.find_signs(img)
        # combine with previous results (if applicable)
        self.tracker.new_data(dec)
        final_decision = self.tracker.combined_results()
        # return results and output image
        #if dec:
        #    return final_decision, np.dstack([labels[0]*255]*3).astype('uint8')
        return final_decision, draw_img

if __name__ == '__main__':
    SVMCLF()
