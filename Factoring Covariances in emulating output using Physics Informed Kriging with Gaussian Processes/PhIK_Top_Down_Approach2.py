import numpy as np
from sympy import symbols, lambdify, sympify
from numpy.linalg import inv, cond, lstsq, pinv
from scipy.linalg import lu
from scipy.io import loadmat
from scipy.integrate import simps

class PhIK_1D():
    def __init__(self,h=None, space_points=None, param_points=None, output_points=None, k=None, kernel=0):
        
        self.kernels = ['squared_exponential','mean']
        # if kernel not in self.kernels:
        #     print(f'Try using from available kernels.\nDefault is squared_exponential')
        #     print(*self.kernels)
        #     exit()
        
        np.random.seed(2)
        
        self.q = symbols('q')
        
        self.h = [sympify(i) for i in h] if h else [self.q**0, self.q]
            
        self.space_points = space_points
        self.param_points = param_points
        self.y = output_points
        self.kernel = self.kernels[kernel]
        
        # print(f'{self.space_points.shape[0]} set of space points with {self.space_points.shape[1]} points each')
        # print(f'{self.param_points.shape[1]} set of param points with {self.param_points.shape[0]} points each')      
        
        if self.kernel in self.kernels[0]:
            self.k = lambda y1,y2,l : np.exp(((y1-y2)**2)/2*l)
        else:
            self.k = lambda y1,y2,mew : np.sum((y1-mew)*(y2-mew))/(self.y.size-1)
               
    def load_variables(self,mat_file):
        return [loadmat(mat_file[0])[i] for i in mat_file[1:]]
    
    def lambdify_h(self,func):
        return [lambdify([self.q], i, 'numpy') for i in func]
    
    def calc_kernel(self,q1,q2):
        if self.kernel in self.kernels[0]:
            p = self.l
        
        elif self.kernel in self.kernels[1]:
            p = np.mean(self.param_points,axis=0)
        
        k = 1
        for i,j,l in zip(q1,q2,p):
            k *= self.k(i,j,l)
            
        return k
    
    def get_val(self,p1,p2=None):
        if p2 is not None:
            m = np.zeros((p1.shape[0],p2.shape[0]))
            for idx_1, q1 in enumerate(p1):
                for idx_2, q2 in enumerate(p2):
                    k = self.calc_kernel(q1,q2)
                    m[idx_1,idx_2] = k
        else:
            m = np.zeros((p1.shape[0]))
            mew = np.mean(p1,axis=0)
            for i,xi in enumerate(p1):
                k = 1
                for j,x in enumerate(xi):
                    k *= np.sum((x - mew[j])*(x - mew[j]))/(self.y.size-1)
                m[i] = k
                
        return m
    
    def least_sq_fit(self,x, y):
        
        h = self.lambdify_h(self.h)
        A = np.ones((x.size, len(self.h)))
        for i in range(len(self.h)):
            f = h[i]
            A[:,i] = f(x)
        b = lstsq(A,y, rcond = None)[0]
        return b
    
    def lu_inv(self,matrix):
        try:
            p,l,u = lu(matrix, permute_l = False)
            l = np.dot(p,l) 
            l_inv = inv(l)
            u_inv = inv(u)
            matrix = np.dot(u_inv,l_inv)
        except:
            matrix = pinv(matrix)
        
        return matrix
    def fit(self, q_star, x_star = None,length_scale=None):
        
        self.l = length_scale
        
        sigma_sq = self.get_val(self.space_points)
        corr_matrix = self.get_val(self.param_points, q_star)
        Cov_matrix_param  = self.get_val(self.param_points, self.param_points)
         
        noise = np.random.normal(np.mean(Cov_matrix_param),np.std(Cov_matrix_param),Cov_matrix_param.shape)
        Cov_matrix_param += noise
    
        self.Inv_Cov_matrix_param = np.array([i * self.lu_inv(Cov_matrix_param) for i in sigma_sq])
        self.corr_vector = np.array([i * corr_matrix for i in sigma_sq])
        
        
        if x_star is None:
            x_star=self.space_points
            
        y_cap = np.array([[self.predict(idx_space,idx_param,q) for idx_space,xi in enumerate(x_star)] for idx_param, q in enumerate(q_star)])
        return y_cap
    
    def predict(self,idx,idx_p,q_star): 
        h_beta_param = 0 
        
        for idx_2 in range(self.param_points.shape[1]):
            h_beta = self.least_sq_fit(self.param_points[:,idx_2].T, self.y[:,idx])*self.h
            h_beta = sympify(' + '.join(str(i) for i in h_beta))
            h_beta = h_beta.subs({self.q:q_star[idx_2]})
            h_beta_param += h_beta 
            
        h_beta_param /= self.param_points.shape[1]
        
        
        zmg = self.y[:,idx] - h_beta_param
        
        R_inv = self.Inv_Cov_matrix_param[idx,:,:]    
        
        corr_vector = self.corr_vector[idx,:,idx_p].T      
    
        y_cap_scalar = h_beta_param + corr_vector@R_inv@zmg
        
        return y_cap_scalar
    
    def get_loss(self,original,predicted,type='l1'):
        l = original-predicted if type=='l1' else np.square(original-predicted)
        return simps(l)
        



