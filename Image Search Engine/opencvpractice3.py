import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

filePath = 'E:/Temp/Screenshot/images (2).jpg'
n_clusters = 5

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def Watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    fg = cv2.erode(thresh,None,iterations = 2)
    cv2.imshow('fg',fg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    bgt = cv2.dilate(thresh,None,iterations = 3)
    cv2.imshow('bgt',bgt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret,bg = cv2.threshold(bgt,1,128,1)
    cv2.imshow('bg',bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    marker = cv2.add(fg,bg)
    cv2.imshow('marker',marker)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    marker32 = np.int32(marker)
    cv2.imshow('marker32',marker32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.watershed(img,marker32)
    cv2.destroyAllWindows()
    m = cv2.convertScaleAbs(marker32)
    cv2.imshow('m',m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ret,thresh = cv2.threshold(m,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.imshow('thresh1',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    res = cv2.bitwise_and(img,img,mask = thresh)
    cv2.imshow('res',res)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def ForeGroundExtract(img):
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    
    plt.imshow(img),plt.colorbar(),plt.show()

if __name__ == '__main__':
    
    img = cv2.imread(filePath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = gray.reshape((gray.shape[0] * gray.shape[1], 3))
    clt = KMeans(n_clusters = n_clusters)
    clt.fit(gray)
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    
    Watershed(img)

