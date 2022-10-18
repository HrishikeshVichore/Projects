from scipy.spatial import distance_matrix as dismat
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import sys
from sklearn.preprocessing import normalize
from scipy.optimize import minimize, Bounds
import pickle 
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

'''make a utils file for all the imp funcs'''
class lengthscale():
    def __init__(self, space,params,output):
        self.y = output
        self.x = space
        self.q = params
        ''' 
        Step 1. calc bins
            1.a divide data into sample sizes
            1.b get space, param and output pts respectively 
        Step 2. calc l for every bin
        Step 3. calc global l
        Step 4. calc diff between local and global l for each bin
        Step 5. Depending on the diff use app. l
            5.a subsample the bin where diff is larger than a thresh.
        '''
    def sample_bins(self):
        
        pass
       
    def MLE_lengthscale(self):
        def fun(l,args):
            y = self.y.flatten()
            n = len(y)
            zmg = y - np.mean(y)
            '''look for calc of h_beta'''
            A = np.outer(self.x,self.q).flatten()
            R_inv = np.linalg.inv(self.calc_cov(A),l)
            sigma = (zmg.T @ R_inv @ zmg)/n
            log_sigma = np.log(sigma)
            log_det_R = np.log(1/np.linalg.det(R_inv))
            obj = n*log_sigma + log_det_R
            
            return obj
        
        l = minimize(fun,[1.5],method='L-BFGS-B')
        
    
def test_class():
    import statsmodels.api as sm
    data = sm.datasets.get_rdataset(dataname='mcycle', package='MASS').data 
    data = data['accel'].to_numpy()
    data = data.reshape(data.shape[0],1)
    print(data.shape)
    lengthscale(data)
    # print(type(data))
    # print(data)

# test_class()
# lengthscale()