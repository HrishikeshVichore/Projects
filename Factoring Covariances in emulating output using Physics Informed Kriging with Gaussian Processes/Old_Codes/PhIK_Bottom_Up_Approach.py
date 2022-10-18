
import numpy as np
from sympy import symbols, lambdify, sympify, pprint, Poly, log, sqrt
import matplotlib.pyplot as plt
from numpy.linalg import inv, cond, lstsq, pinv, cholesky
from sympy.plotting import plot
from scipy.io import loadmat, savemat

class PhIK_1D():
    def __init__(self,h=None, space_points=None, param_points=None, output_points=None, mat_file=None, k=None):
        np.random.seed(2)
        self.x = symbols('x')
        self.q = symbols('q')
        self.a = symbols('a')
        self.h = [sympify(i) for i in h] if h else [self.q**0, self.q]

        if mat_file:
            m = self.load_variables(mat_file)
            #Select only those arrays that are necessary. Space, param and o/p.
            self.space_points, self.param_points, self.y = m
            print(self.space_points)
            print(self.param_points)
            exit()
            
        else:
            self.space_points = space_points
            self.param_points = param_points
            self.y = output_points
        print(self.space_points.shape)
        print(self.param_points.shape)
        print(self.y.shape)
        exit()      
        self.mew_space_poins = np.mean(self.space_points)
        self.mew_param_points = np.mean(self.param_points)
        self.k = k if k else lambda y1,mew1,y2,mew2 : np.sum((y1 - mew1)*(y2-mew2))/(self.y.size-1)
    
    def load_variables(self,mat_file):
        return [loadmat(mat_file[0])[i] for i in mat_file[1:]]
    
    def lambdify_h(self,func):
        return [lambdify([self.q], i, 'numpy') for i in func]
    
    
    def least_sq_fit(self,x, y):
        
        h = self.lambdify_h(self.h)
        A = np.ones((x.size, len(self.h)))
        print(x.shape)
        exit()
        for i in range(len(self.h)):
            f = h[i]
            A[:,i] = f(x)
        b = lstsq(A,y, rcond = None)[0]
        return b
    
    def fit(self, q_star, x_star = None):
        
        Cov_matrix_param = np.array([[self.k(i,self.mew_param_points, j,self.mew_param_points) for j in self.param_points] for i in self.param_points])
        noise = np.random.normal(np.mean(Cov_matrix_param),np.std(Cov_matrix_param),Cov_matrix_param.shape)
        Cov_matrix_param += noise 
        try:
            self.Inv_Cov_matrix_param = inv(Cov_matrix_param)
        except:
            self.Inv_Cov_matrix_param = pinv(Cov_matrix_param)
        if x_star is None:
            x_star=self.space_points[0]
        y_cap = np.array([[self.predict(idx,xi,q) for idx, xi in enumerate(x_star)] for q in q_star])
        return y_cap
    
    def predict(self,idx,xi,q_star):  
        
        h_beta = self.least_sq_fit(self.param_points.T, self.y[:,idx])*self.h
        h_beta = sympify(' + '.join(str(i) for i in h_beta))
        h_beta = h_beta.subs({self.q:q_star})
        zmg = self.y[:,idx] - h_beta
        
        sigma_sq = self.k(xi,self.mew_space_poins,xi,self.mew_space_poins)
        R_inv = self.Inv_Cov_matrix_param*sigma_sq
        param_points = np.append(self.param_points,[[q_star]],axis = 0)
        new_mean = np.mean(param_points)
        corr_vector = sigma_sq*np.array([self.k(i,self.mew_param_points, q_star,new_mean) for i in self.param_points]).T        
        y_cap_scalar = h_beta + corr_vector@R_inv@zmg
        # print(y_cap_scalar)
        return y_cap_scalar

P = PhIK_1D(mat_file=['train','zu', 'phi0', 'u'])
x_star, q_star = P.load_variables(['test.mat','test_zu', 'test_phi0'])
y_cap = P.fit(q_star[0], x_star=None)[0]
y_cap = np.array([np.float32(i) for i in y_cap])

print(y_cap.T)
savemat('y_cap8.mat',{'y_cap':y_cap})
print(y_cap.shape)


