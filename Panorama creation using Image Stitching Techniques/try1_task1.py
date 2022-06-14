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
np.random.seed(1)
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random

def euclidean_distance(x,y):
    return np.sqrt(np.sum(np.power(x-y,2),axis = 1))

def get_keypoint_matches(left,right,n_points,ratio=0.75):
    
    (keyPoints_left,des_left) = left
    (keyPoints_right,des_right) = right
    
    kps = [i.pt for i in keyPoints_left]
    kpd = [i.pt for i in keyPoints_right]
    
    des_dict_left = {kps[i]:des_left[i] for i in range(n_points)}
    des_dict_right = {kpd[i]:des_right[i] for i in range(n_points)}
    
    dist = {key:euclidean_distance(val, des_left) for key,val in des_dict_right.items()}        
    
    rawMatches = {p:sorted(enumerate(q), key=lambda x: x[1])[:2] for p,q in dist.items()}
    matches = {p:q[0] for p,q in rawMatches.items() if q[0][1]<ratio*q[1][1]}
    matches = {p:(kps[q[0]],q[1]) for p,q in matches.items()}
    kps = np.array(list(matches.keys())) # Left Image
    kpd = np.array([i[0] for i in matches.values()]) #Right Image 
    return kps,kpd

def fit_fn(kps,kpd):
    A = np.zeros((2*len(kps),9))
    A[::2, 2] = 1
    A[1::2, 5] = 1
    A[::2,[0,1]] = kps
    A[1::2, [3,4]] = kps
    A[::2, -1] = -kpd[:,0]
    A[1::2, -1] = -kpd[:,1]
    A[::2, [6,7]] = -np.multiply(kps, kpd.T[0][:, np.newaxis])
    A[1::2, [6,7]] = -np.multiply(kps, kpd.T[1][:, np.newaxis])
    _,_,v = np.linalg.svd(A, full_matrices = True)
    
    x = v[-1] # Selecting last row of V.T for minimization.
    x = x/x[-1]
    
    x = np.array(x).reshape(3,3)
    
    l = np.sqrt(1/sum([x[2,i]**2 for i in range(2)])) # Lambda
      
    homographyMatrix = l*x

    return homographyMatrix

def evaluate_model(kps,kpd,H,threshold):
    kps_aug = np.concatenate((kps,np.ones((kps.shape[0],1))), axis = 1)
    kpd_aug = np.concatenate((kpd,np.ones((kpd.shape[0],1))), axis = 1)
    
    kpd_pred = [H@i for i in kps_aug]
    kpd_pred = np.array([i/i[-1] for i in kpd_pred])
    dist = euclidean_distance(kpd_pred, kpd_aug)
    index = np.where(dist<threshold)
    kps_inline = kps[index]
    kpd_inline = kpd[index]
    performance = (len(kps_inline)/len(kps))*100
    
    return kps_inline, kpd_inline, performance
    


def ransac(kps,kpd,threshold,samples = 4,n_iterations = 50):
    best_kps_inlier = None
    best_kpd_inlier = None
    best_performance = 0
    
    for _ in range(n_iterations):
        index = np.random.choice(kps.shape[0], samples, replace = False)
        H = fit_fn(kps[index], kpd[index])
        kps_inlier, kpd_inlier, performance = evaluate_model(kps, kpd, H, threshold)
        
        if performance > best_performance:
            best_performance = performance
            best_kps_inlier = kps_inlier
            best_kpd_inlier = kpd_inlier  
    H = fit_fn(best_kps_inlier, best_kpd_inlier)        
    return H     

def get_warped_H(left_img, right_img, H):
    (w1, h1), (w2, h2) = left_img.shape[:2], right_img.shape[:2]
    
    left_img_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    
    right_img_dims = cv2.perspectiveTransform(np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2), H)
    result_img_dims = np.concatenate((left_img_dims, right_img_dims), axis=0)

    [x_min, y_min] = np.int32(result_img_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_img_dims.max(axis=0).ravel() + 0.5)

    offset = np.array([[1, 0, -x_min],[0, 1, -y_min],[0, 0, 1]])
    H = H@offset
    result_img_dims = (x_max-x_min, y_max-y_min)

    result_img = cv2.warpPerspective(right_img, H,result_img_dims)
    result_img[-y_min:w1+-y_min, -x_min:h1+-x_min] = left_img
    
    return result_img

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """
    n_points = 4000
    sift = cv2.SIFT.create(n_points)
    
    left = sift.detectAndCompute(left_img,None)
    right = sift.detectAndCompute(right_img,None)
    
    kps,kpd = get_keypoint_matches(left,right,n_points,ratio = 0.55)
    
    H = ransac(kps, kpd, threshold = 5, samples = 4, n_iterations = 3000)
    
    result_img = get_warped_H(left_img, right_img, H)
    
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
