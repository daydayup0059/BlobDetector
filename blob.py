#!/usr/bin/python
#斑点检测  https://blog.csdn.net/u014072827/article/details/111033547
# Standard imports
import cv2
import numpy as np

# Read image  cv2.IMREAD_GRAYSCALE,按灰度模式读取图像
im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters. 设置SimpleBlobDetector参数
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10  #二值化的起始阈值
params.maxThreshold = 200  #二值化的终止阈值


# Filter by Area.
params.filterByArea = True  #斑点面积的限制变量
params.minArea = 1500  #斑点的最小面积

# Filter by Circularity 通过圆率来过滤>0.1
params.filterByCircularity = True #//斑点圆度的限制变量，默认是不限制
params.minCircularity = 0.1     #斑点的最小圆度

# Filter by Convexity  //斑点凸度的限制变量
params.filterByConvexity = True
params.minConvexity = 0.87  #//斑点的最小凸度
    
# Filter by Inertia
# 通过惯性比滤波进行过滤
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
#不同的cv2版本函数名称不同，4.0以上是SimpleBlobDetector_create
if int(ver[0]) < 3 :
	#创建一个检测器并使用默认参数
	detector = cv2.SimpleBlobDetector(params)
else :
	#
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs. 检测blobs
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

