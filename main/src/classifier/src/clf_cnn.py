#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int8, Time
import os

import numpy as np
import cv2
from sklearn import svm
from scipy.ndimage.measurements import label
import imagefunctions
from tracker import Tracker
import pickle
#import random
import time

###########################################################3
# sliding window parameters
imgsize = (128,96)
windowsize = (50,50)
slidestep = (5,5) # number of pixels to slide window
min_red_pixels = 150 # min red pixel to process window

class SVMCLF():
    def __init__(self):

        # ROS nodes
        rospy.init_node('classifier', anonymous=True)
        rospy.Subscriber("camera/image", Image, self.callback, queue_size=1)
        self.pub = rospy.Publisher('driver_node/drivestate', Int8, queue_size=1, latch=True)
        self.imgcnn_pub = rospy.Publisher('camera/imgcnn', Image, queue_size=1)

        # publish "stop" on the drivestate topic
        print 'publishing -1'
	self.pub.publish(-1)
	print 'Importing tensorflow ...'
        import tensorflow as tf
        print 'Tensorflow imported.'
        # publish "go" on the drivestate topic

        # Trained SVM model
        fn_model = rospy.get_param('~cnnModelFile')
        graph_file = rospy.get_param('~cnnGraphFile')
        saver = tf.train.import_meta_graph(graph_file)
	self.sess = tf.Session()
	graph_dir = os.path.dirname(graph_file)
	graph = tf.get_default_graph()
	saver.restore(self.sess, tf.train.latest_checkpoint(graph_dir))
	self.prediction_op = graph.get_tensor_by_name("prediction_op:0")
	self.logits_op = graph.get_tensor_by_name("logits_op:0")
	#self.probs = graph.get_tensor_by_name("probs:0")
	self.x_placeholder = graph.get_tensor_by_name("x:0")
	self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
	# svmparams = pickle.load(open(fn_params, 'rb')) #pickle.load(f2)
        # self.fmean = svmparams['fmean']
        # self.fstd = svmparams['fstd']

        # Settings
        self.nhistory = 1 # tracker buffer
        self.dt = 0.1 # update interal
        self.K_detthresh_stop = 0.7 # detection threshold (factor of window count)
        self.K_detthresh_warn = 1.2 # detection threshold (factor of window count)
        self.K_mapthresh = 0.08 # discard threshold (factor of window count)
        self.K_stopbias = 0.4 # bias to favor STOP over WARN (factor of window count)

        # Updates
        self.drive_state = 0
        self.img = [None]
        self.lastimgtime = -1

        # tracker
        self.tracker = Tracker(self.nhistory)

        # start
        self.loop()

    def callback(self,rosimg):
        self.img = CvBridge().imgmsg_to_cv2(rosimg)
        self.lastimgtime = rosimg.header.stamp.secs

    def loop(self):
        rate = rospy.Rate(1/self.dt)
        lastupdate = -1
        while not rospy.is_shutdown():
            if (self.lastimgtime != lastupdate):
                start_time = time.time()
                # ---- process frame ---
                rosimg = self.img # copy to local memory before processing
                dec,draw_img = self.processOneFrame(rosimg)
                lastupdate = self.lastimgtime
                # Publish results
                #print "State=" + str(dec) + " (" + str(time.time() - start_time) + "s)"
                self.drive_state = dec
                #print ("Drivestate = ",dec)
                self.pub.publish(self.drive_state)
                # Optional - Publish image for monitoring
                self.imgcnn_pub.publish(CvBridge().cv2_to_imgmsg(draw_img, "bgr8"))

        rate.sleep()

    #def getFeatures(self,img):
    #    return [
    #        imagefunctions.num_corners(img),
    #        imagefunctions.num_edges(img),
    #        imagefunctions.num_red_pixels(img),
    #        imagefunctions.num_white_pixels(img),
    #        imagefunctions.abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(100, 200)),
    #        imagefunctions.mag_thresh(img, sobel_kernel=5, mag_thresh=(100, 180)),
    #        imagefunctions.dir_threshold(img, sobel_kernel=3, thresh=(np.pi/8, np.pi/4))
    #    ]

    #def normalize_features(self,feature_vector,fmn,fsd):
    #    numDim = len(feature_vector)
    #    normFeatures = []
    #    normfeat = [None]*numDim
    #    for i in range(numDim):
    #        normfeat[i] = (feature_vector[i]-fmn[i])/fsd[i]
    #    normFeatures.append(normfeat)
    #    #transpose result
    #    res = np.array(normFeatures).T
    #    return res

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
            #feat = self.getFeatures(test_img)
            # normalize features
            #normfeat = self.normalize_features(feat,self.fmean,self.fstd)
            # assemble batch
            #testvec = np.asarray(normfeat).reshape(1,-1)
            fvec.append(test_img)

        # batch prediction
        if (np.array(fvec).ndim >= 3):
	    #prob_vec = self.sess.run(self.probs, feed_dict={self.x_placeholder:np.asarray(fvec, dtype=np.float32), self.keep_prob:1.0})
            rvec = self.sess.run(self.prediction_op, feed_dict={self.x_placeholder:np.asarray(fvec, dtype=np.float32), self.keep_prob: 1.0}) #np.array(fvec).squeeze(axis=1))
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
        startx = 10
        stopx = 68 #imgsize[0]-windowsize[0] #80
        starty = 0 #20 #19
        stopy = imgsize[1]-windowsize[1] #30

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
            cv2.rectangle(img,(startx, starty),(endx, endy),(200,0,0),1)
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
            cv2.rectangle(img,(startx, starty),(endx, endy),(0,0,255),1)

        score_stop = np.max(heat_stop)
        score_warn = np.max(heat_warn)
        print '[scores] stop:' + str(score_stop) + ' warn:' + str(score_warn)

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
        #stopbias = self.K_stopbias * numwin
        print 'numwin = ' + str(numwin) + ', detthreshSTOP = ' + str(detthresh_stop) + ', detthreshWARN = ' + str(detthresh_warn)
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
        return final_decision, draw_img

if __name__ == '__main__':
    SVMCLF()
