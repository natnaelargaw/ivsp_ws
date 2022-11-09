#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('ivsp')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

  
  """ callback function for the topic /naoqi_driver/camera/front/image_raw  """
  def saliency_model_callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, ((cols/2)-15,(rows/2)-15), 30, 255)

      cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)
      """ prediction and locomotion command is acquired after this function call """
      saliency_map = self.predict(cv_image)

      [X, Y, Z] = self.localize_torsow(saliency_map)

      
  
  """ Video saliency prediction model """
  def predict(self, image_data):
    """
    load model here  
    predict
    """
    predicted_map = image_data
    try:
        self.saliency_map_pub.publish(self.bridge.cv2_to_imgmsg(predicted_map, "bgr8"))
        print(predicted_map.shape)
    except CvBridgeError as e:
        print(e)

    return predicted_map


  """ calculates x,y,z value of all dof for the robot torsow and publishes both to the
   torsow subscriber and a separate topic called /saliency/salient_roi 
  """
  def localize_torsow(self, saliency_map):
    hello_coordinte = "hello [X, Y, Z]%s   this is the coordinate" % rospy.get_time()
    self.localize_pub.publish(hello_coordinte)

    return ["X", "Y", "Z"]


""" entry function for the video saliency prediction model"""
def main(args):
  ic = dynamic_saliency_prediction()
  rospy.init_node('dynamic_saliency_prediction', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

""" entry point for the file """
if __name__ == '__main__':
    main(sys.argv)
