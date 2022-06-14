"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

def euclidean_distance(x,y):
    return np.sqrt(np.sum(np.power(x-y,2),axis = 1))

def get_knn(dist,n=2,ratio=0.75):
    rawMatches = []
    neigh_ind = []
    
    for row in dist:
        sorted_neigh = sorted(enumerate(row), key=lambda x: x[1])[:2]
        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]
    
        rawMatches.append(dist_list)
        neigh_ind.append(ind_list)
    
    matches = []
    for m,n in rawMatches:
        if m < n*ratio:
            matches.append((m,n))
    
    
    
    return neigh_ind,matches

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    sift = cv2.SIFT.create(10000)
    (keyPoints_right,des_right) = sift.detectAndCompute(right_img,None)
    (keyPoints_left,des_left) = sift.detectAndCompute(left_img,None)
    
    # print(len(des_right),len(des_left))
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = bf.knnMatch(des_left,des_right,2)
    matches = []
    ratio = 0.9
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
            
    src_pts = np.float32([ keyPoints_left[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keyPoints_right[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    result_img = cv2.warpPerspective(left_img, H, ((left_img.shape[1] + right_img.shape[1]), right_img.shape[0])) #wraped image
    result_img[0:right_img.shape[0], 0:right_img.shape[1]] = right_img
    #result_img[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    # cv2.imshow('result', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
    #raise NotImplementedError
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


