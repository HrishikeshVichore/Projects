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
        self.h = [sympify(i) for i in h] if h else lambda space=1 : [self.x**0, self.x] if space else [self.q**0, self.q]

        if mat_file:
            m = self.load_variables(mat_file)
            #Select only those arrays that are necessary. Space, param and o/p.
            self.space_points, self.param_points, self.y = m
            
        else:
            self.space_points = space_points
            self.param_points = param_points
            self.y = output_points
              
        self.mew_space_poins = np.mean(self.space_points)
        self.mew_param_points = np.mean(self.param_points) 
        self.k = k if k else lambda y1,mew1,y2,mew2 : np.sum((y1 - mew1)*(y2-mew2))/(self.y.size-1)
    
    def load_variables(self,mat_file):
        return [loadmat(mat_file[0])[i] for i in mat_file[1:]]
    
    def lambdify_h(self,var,func):
        return [lambdify(var, i, 'numpy') for i in func]
    
    
    def find_coeff(self,points, is_space=0):
        var = self.x if is_space else self.q
        h = self.lambdify_h(var, self.h(is_space))
        A = np.ones((points.size, len(self.h(is_space))))
            
        for i in range(len(self.h(is_space))):
            f = h[i]
            A[:,i] = [f(i[0]) for i in points]    
            
        R_inv = self.Inv_Cov_matrix_space if is_space else self.Inv_Cov_matrix_param
        
        beta = inv(A.T@R_inv@A)@A.T@self.Inv_Cov_matrix_param@self.y
        
        h_beta = self.h(is_space) @ beta 
        
        return h_beta
    
    def fit(self, q_star, x_star = None):
        
        Cov_matrix_param = np.array([[self.k(i,self.mew_param_points, j,self.mew_param_points) for j in self.param_points] for i in self.param_points])
        # Cov_matrix_param = np.array([[self.k(1,1,1,1)]])
        noise = np.random.normal(np.mean(Cov_matrix_param),np.std(Cov_matrix_param),Cov_matrix_param.shape)
        Cov_matrix_param += noise 
        # self.h_beta = np.mean([np.multiply(self.least_sq_fit(self.param_points.T, self.y[:,i]),self.h(0)) for i in range(self.y.shape[1])])
              
        
        try:
            self.Inv_Cov_matrix_param = inv(Cov_matrix_param)
        
        except:
            self.Inv_Cov_matrix_param = pinv(Cov_matrix_param)
        
        self.h_beta = self.find_coeff(self.param_points)
        # print(self.h_beta)
        
        if x_star is None:
            x_star=self.space_points[0]
            
        y_cap = np.array([[self.predict(idx,xi,q) for idx, xi in enumerate(x_star)] for q in q_star])
        # print(y_cap.shape)    
        return y_cap
    
    def predict(self,idx,xi,q_star):  
        
        # print(idx,xi)
        h_beta = self.h_beta[idx].subs({self.q:q_star})
        # print(h_beta)
        
        R_inv = self.Inv_Cov_matrix_param*(1/xi)
        
        param_points = np.append(self.param_points,[[q_star]],axis = 0)
        new_mean = np.mean(param_points)
        # print(new_mean, self.mew_param_points)
        corr_vector = np.array([self.k(i,self.mew_param_points, q_star,new_mean) for i in self.param_points]).T
        # print(f'corr_vector shape = {corr_vector.shape}')
        
        zmg = self.y[:,idx] - h_beta
        
        y_cap_scalar = h_beta + corr_vector@R_inv@zmg
        
        return y_cap_scalar
    
    # def find_loss(self, y_cap, test_file):
    #     y_original = loadmat(test_file)['k'][0]
    #     loss = sqrt(np.square(np.subtract(y_original,y_cap)).mean())
    #     print(loss)
    #     # print(y_original)
    #     # print('\nAccuracy:' + np.sqrt(np.square(np.subtract(y_original,y_cap)).mean()) + '%')
    #

P = PhIK_1D(mat_file=['train','zu', 'phi0', 'u'])
x_star, q_star = P.load_variables(['test.mat','test_zu', 'test_phi0'])
# print(x_star, q_star)
y_cap = P.fit(q_star[0], x_star=None)
y_cap = np.array([np.float32(i) for i in y_cap])
savemat('y_cap6_copy.mat',{'y_cap6_copy':y_cap})
print(y_cap.shape)
# loss = P.find_loss(y_cap,test_file='test')


