import numpy as np
from sympy import symbols, lambdify, sympify
from numpy.linalg import inv, cond, lstsq, pinv
from scipy.linalg import lu
from scipy.io import loadmat

class PhIK_1D():
    def __init__(self,h=None, space_points=None, param_points=None, output_points=None, k=None, kernel='squared_exponential'):
        self.kernels = ['squared_exponential','mean']
        if kernel not in self.kernels:
            print(f'Try using from available kernels.\nDefault is squared_exponential')
            print(*self.kernels)
            exit()
        
        np.random.seed(2)
        
        self.q = symbols('q')
        
        self.h = [sympify(i) for i in h] if h else [self.q**0, self.q]
            
        self.space_points = space_points
        self.param_points = param_points
        self.y = output_points
        self.kernel = kernel
        
        print(f'{self.space_points.shape[0]} set of space points with {self.space_points.shape[1]} points each')
        print(f'{self.param_points.shape[1]} set of param points with {self.param_points.shape[0]} points each')      
                
        self.k = lambda y1,y2,mew : np.sum((y1-mew)*(y2-mew))/(self.y.size-1) if self.kernel in self.kernels[1] else lambda y1,y2,l : np.exp(((y1-y2)**2)/2*l)
    
    def load_variables(self,mat_file):
        return [loadmat(mat_file[0])[i] for i in mat_file[1:]]
    
    def lambdify_h(self,func):
        return [lambdify([self.q], i, 'numpy') for i in func]
    
    
    def least_sq_fit(self,x, y):
        
        h = self.lambdify_h(self.h)
        A = np.ones((x.size, len(self.h)))
        for i in range(len(self.h)):
            f = h[i]
            A[:,i] = f(x)
        b = lstsq(A,y, rcond = None)[0]
        return b
    def calc_kernel(self,q1,q2):
        if self.kernel in self.kernels[0]:
            p = self.l
        
        elif self.kernel in self.kernels[1]:
            p = np.mean(self.param_points,axis=0)
        print(p)
        k = 1
        for i,j,l in zip(q1,q2,p):
            k *= self.k(i,j,l)
            
            
        return k
    
    def fit(self, q_star, x_star = None, length_scale=None):
        self.l = length_scale
        Cov_matrix_param  = np.zeros((self.param_points.shape[0],self.param_points.shape[0]))
        for idx_1, q1 in enumerate(self.param_points):
            for idx_2, q2 in enumerate(self.param_points):
                k = self.calc_kernel(q1,q2)
                Cov_matrix_param[idx_1,idx_2] = k
        
        
        noise = np.random.normal(np.mean(Cov_matrix_param),np.std(Cov_matrix_param),Cov_matrix_param.shape)
        Cov_matrix_param += noise         
        
        try:
            self.Inv_Cov_matrix_param = inv(Cov_matrix_param)
        except:
            self.Inv_Cov_matrix_param = pinv(Cov_matrix_param)
            
        
        if x_star is None:
            x_star=self.space_points[0]
            
        y_cap = np.array([[self.predict(idx_space,xi,q) for idx_space,xi in enumerate(x_star)] for q in q_star])
        return y_cap
    
    def predict(self,idx,xi,q_star): 
        h_beta_param = 0 
        
        for idx_2 in range(self.param_points.shape[1]):
            h_beta = self.least_sq_fit(self.param_points[:,idx_2].T, self.y[:,idx])*self.h
            h_beta = sympify(' + '.join(str(i) for i in h_beta))
            h_beta = h_beta.subs({self.q:q_star[idx_2]})
            h_beta_param += h_beta 
            
        zmg = self.y[:,idx] - h_beta_param
        corr_vector = np.zeros((self.param_points.shape[0],1))
        
        for idx_1, q1 in enumerate(self.param_points):
            k = self.calc_kernel(q1,q_star)
            corr_vector[idx_1] = k
        
        mew_space = np.mean(self.space_points)    
        s = lambda xi : np.sum((xi - mew_space)*(xi - mew_space))/(self.y.size-1)
        sigma_sq = s(xi)
        # sigma_sq = (zmg.T@self.Inv_Cov_matrix_param@zmg)/self.param_points.size
        R_inv = self.Inv_Cov_matrix_param*sigma_sq    
        corr_vector = corr_vector.T*sigma_sq   
        # print(corr_vector.shape)
        # print(R_inv.shape)
        # print(zmg.shape)  
        y_cap_scalar = h_beta_param + corr_vector@R_inv@zmg
        
        return y_cap_scalar



