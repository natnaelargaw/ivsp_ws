#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import roslib
roslib.load_manifest('ivsp')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import os
import numpy as np
from config import *
from model import *
from utility import *

from PIL import Image as im
from numpy import asarray


"""
This class subscribes to any live video stream, /naoqi_driver/camera/front/image_raw topic in this case, and 
predicts the saliency map and set locomotion parameters for the robot pepper in real-time. 

The class is initialized with cv_bridge instance to convert cv image to ros image and various publisher and subscriber instances
along with their call back. 

The saliency model callback function subscribes to /naoqi_driver/camera/front/image_raw topic and perform two major tasks. 
predicting the saliency map using predict() function and localizing the ROI of the robot using localize_torsow().

"""

class dynamic_saliency_prediction:
  
  def __init__(self):
    self.saliency_map_pub = rospy.Publisher("saliency/saliency_image",Image, queue_size=10)
    self.localize_pub = rospy.Publisher("saliency/salient_roi",String,queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image, self.saliency_model_callback)

    self.x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.x2 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.x3 = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    self.stateful = True
    self.m = Model(inputs=[self.x, self.x2, self.x3], outputs=transform_saliency([self.x, self.x2, self.x3], self.stateful))
    print("Loading MODS weights")
    self.m.load_weights('Saliencyconvlstm.h5') 
    self.queue = deque()


  """ callback function for the topic /naoqi_driver/camera/front/image_raw  """
  def saliency_model_callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
   
    # get video access
    if len(self.queue) < num_frames:
      self.queue.append(cv_image)
    else:
      self.queue.popleft()
      self.queue.append(cv_image)

      Xims = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality
      Xims2 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality
      Xims3 = np.zeros((1, len(self.queue), shape_r, shape_c, 3))  # change dimensionality

      [X, X2, X3] = preprocess_images_realtime(self.queue, shape_r, shape_c)
      # print(X.shape, "X shape new")
      #   # print("Inside Test Generator ", X.shape)
      Xims[0] = np.copy(X)
      Xims2[0] = np.copy(X2)
      Xims3[0] = np.copy(X3)

      prediction = self.m.predict([Xims,Xims2,Xims3])
      # print("Prediction shape: ", prediction.shape)

      for j in range(len(self.queue)):
        orignal_image = self.queue[0]
        print(orignal_image.shape, "Queue shape")
        x, y = divmod(j, len(self.queue))

      #print(prediction[0,0,:,:,0].shape)
      self.publish_prediction(prediction[0,0,:,:,0])
      self.m.reset_states()
  
  """ Moving object detection and segmentation model """
  def publish_prediction(self, image_data):
    """
    publish mods map here
    """
    # print("Publishable data received")
    predicted_map = image_data
    try:
        self.saliency_map_pub.publish(self.bridge.cv2_to_imgmsg(predicted_map, "32FC1"))
        print(predicted_map.shape, " MODS map published")
    except CvBridgeError as e:
        print(e)

    # localize_torsow(predicted_map)

  """ calculates x,y,z value of all dof for the robot torsow and publishes both to the
   torsow subscriber and a separate topic called /mods/object_roi 
  """
  # robot head movement setter. under development
  def localize_torsow(self, object_map):
    hello_coordinte = "hello [X, Y, Z]%s   this is the coordinate" % rospy.get_time()
    self.localize_pub.publish(hello_coordinte)

""" entry function for the video saliency prediction model"""
def main(args):
  ic = dynamic_saliency_prediction()
  rospy.init_node('dynamic_saliency_prediction', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv2.destroyAllWindows()

""" entry point for the file """
if __name__ == '__main__':
    main(sys.argv)

