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

def dont_use(result_img):
    gray = cv2.cvtColor(result_img,cv2.COLOR_BGR2GRAY)

    # threshold
    _,thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    
    # apply close and open morphology to fill tiny black and white holes
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # get contours (presumably just one around the nonzero pixels) 
    # then crop it to bounding rectangle
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        crop = result_img[y:y+h,x:x+w]
        # show cropped image
    result_img = crop
    return result_img

def get_imgOffset(base_img, warped_img, window):
    min1 = np.inf
    for x in range(-window,window):
        for y in range(-window,window):
            temp_img= np.roll(warped_img, [x,y])
            ssd = np.sum(np.power((base_img - temp_img),2))
            if ssd < min1:
                min1 = ssd
                offset_x = x
                offset_y = y
    return offset_x, offset_y             


def euclidean_distance(x,y):
    return np.sqrt(np.sum(np.power(x-y,2),axis = 1))

def get_keypoint_matches(kps,kpd,ratio=0.75):
    dist = {tuple(x):euclidean_distance(x, kpd) for x in kps}        
    rawMatches = {p:sorted(enumerate(q), key=lambda x: x[1])[:2] for p,q in dist.items()}
    matches = {p:q[0] for p,q in rawMatches.items() if q[0][1]<ratio*q[1][1]}
    matches = {p:(kpd[q[0]],q[1]) for p,q in matches.items()}
    # print(len(matches))
    # print(next(iter((matches.items())) ))
    return matches

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
    
    l = np.sqrt(1/sum([x[2,i]**2 for i in range(3)])) # Lambda
      
    homographyMatrix = l*x
    # homographyMatrix = homographyMatrix.reshape((9,))
    # print(l,homographyMatrix)
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
        
    print(len(best_kps_inlier), best_performance)  
    H = fit_fn(best_kps_inlier, best_kpd_inlier)        
    return H        

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    sift = cv2.SIFT.create(4000)
    (keyPoints_right,des_right) = sift.detectAndCompute(right_img,None)
    (keyPoints_left,des_left) = sift.detectAndCompute(left_img,None)
    
    kps = np.array([i.pt for i in keyPoints_left])
    kpd = np.array([i.pt for i in keyPoints_right])
    
    matches = get_keypoint_matches(kps,kpd,ratio = 0.85)
    
    kps = np.array(list(matches.keys())) # Left Image
    kpd = np.array([i[0] for i in matches.values()]) #Right Image 
    # threshold = np.mean([i[1] for i in matches.values()]) #Average Distance is threshold
    # print(threshold)
    np.random.seed(1)
    #samples = int(np.ceil(0.1*len(kps))) # 10% of total data as samples
    H = ransac(kps, kpd, threshold = 6, samples = 4, n_iterations = 500)
    result_img = cv2.warpPerspective(right_img, H, ((left_img.shape[1] + right_img.shape[1]), left_img.shape[0])) #wraped image

    # now paste them together
    #result_img[0:right_img.shape[0], 0:right_img.shape[1]] = right_img
    #result_img[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    #exit()
    #raise NotImplementedError
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)


