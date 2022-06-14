import numpy as np
from sympy import symbols, lambdify, sympify, pprint, Poly, log
import matplotlib.pyplot as plt
from numpy.linalg import inv, cond, lstsq, pinv, cholesky
from sympy.plotting import plot
from scipy.io import loadmat
# from PhIK_Try3 import PhIK as try3

class PhIK_1D():
    def __init__(self,h=None, space_points=None, param_points=None, output_points=None, mat_file=None, k=None):
        np.random.seed(2)
        self.x = symbols('x')
        self.q = symbols('q')
        self.a = symbols('a')
        self.h = [sympify(i) for i in h] if h else lambda space=1 : [self.x**0, self.x] if space else [self.q**0, self.q]

        if mat_file:
            #Select only those arrays that are necessary. Space, param and o/p.
            m = [loadmat(mat_file[0])[i] for i in mat_file[1:]]
            self.space_points, self.param_points, self.y = m
        else:
            self.space_points = space_points
            self.param_points = param_points
            self.y = output_points
            
        self.mew_space_poins = np.mean(self.y, axis = 1)
        self.mew_param_points = np.mean(self.param_points) 
        self.k = k if k else lambda y1,mew1,y2,mew2 : np.sum((y1 - mew1)*(y2-mew2))/(self.y.size-1)
    
    def lambdify_h(self,func):
        return [lambdify([self.x], i, 'numpy') for i in func]
    
    
    def least_sq_fit(self,x, y):
        h = self.lambdify_h(self.h())
        A = np.ones((x.size, len(self.h())))
        for i in range(len(self.h())):
            f = h[i]
            A[:,i] = f(x)
        b = lstsq(A,y, rcond = None)[0]
        return b
    
    def fit(self):
        h_beta_space = np.mean([np.multiply(self.least_sq_fit(self.space_points, self.y[i,:]),self.h()) for i in range(self.y.shape[0])])
        h_beta_param = np.mean([np.multiply(self.least_sq_fit(self.param_points.T, self.y[:,i]),self.h(0)) for i in range(self.y.shape[1])])
        self.h_beta = h_beta_param + h_beta_space # Works as an OR operation
        # print(h_beta_space, h_beta_param)
        # print(self.h_beta)
        
        '''[(y-u)*(y-u)' + Noise*I]'''
        Cov_matrix_space = np.array([self.k(self.y[i,:],self.mew_space_poins[i], self.y[i,:],self.mew_space_poins[i]) for i in range(len(self.mew_space_poins))])
        noise = np.random.normal(np.mean(Cov_matrix_space),np.std(Cov_matrix_space),Cov_matrix_space.shape)
        Cov_matrix_space += noise 
        # print(Cov_matrix_space, Cov_matrix_space.shape)
        Inv_Cov_matrix_space = np.reciprocal(Cov_matrix_space)
        
        Cov_matrix_param = np.array([[self.k(i,self.mew_param_points, j,self.mew_param_points) for j in self.param_points] for i in self.param_points])
        noise = np.random.normal(np.mean(Cov_matrix_param),np.std(Cov_matrix_param),Cov_matrix_param.shape)
        Cov_matrix_param += noise 
        
        '''
        Rewrite cov_func for cholesky decompostion.
        '''
        # print(Cov_matrix_param)
        # L = cholesky(Cov_matrix_param)
        # P = inv(L).T
        # Inv_Cov_matrix_param = P.T @ P
        
        # print(Cov_matrix_param.shape)
        
        try:
            # Inv_Cov_matrix_space = inv(Cov_matrix_space)
            Inv_Cov_matrix_param = inv(Cov_matrix_param)
        
        except:
            # Inv_Cov_matrix_space = pinv(Cov_matrix_space)
            Inv_Cov_matrix_param = pinv(Cov_matrix_param)


        # print(Inv_Cov_matrix_space)
        # print(f'Condition number of Covariance Matrix Space = {round(cond(Inv_Cov_matrix_space),3)}')
        # print(f'Condition number of Covariance Matrix Param = {round(cond(Inv_Cov_matrix_param),3)}')
        
        # self.R_inv = np.kron(Inv_Cov_matrix_space, Inv_Cov_matrix_param)
        # self.R_inv = np.reshape(self.R_inv, (5,5,5))
        # print(self.R_inv)
        print('a')
        self.R_inv = np.array([i*Inv_Cov_matrix_param for i in Inv_Cov_matrix_space])
        print(self.R_inv, self.R_inv.shape)
        # print(cond(self.R_inv), self.R_inv.shape)
        
    def predict(self,x_star, q_star):
        # print(self.h_beta)
        self.h_beta = self.h_beta.subs({self.x:x_star,self.q:q_star})
        # print(self.h_beta)
        zmg = self.y - self.h_beta
        print(zmg.shape)
        param_points = np.append(self.param_points,[[q_star]],axis = 0)
        new_mean = np.mean(param_points)
        # print(new_mean, self.mew_param_points)
        corr_vector = np.array([self.k(i,self.mew_param_points, q_star,new_mean) for i in self.param_points])
        
        print(corr_vector.shape)
        # self.param_points = param_points
        # pprint(zmg)
        y_cap = self.h_beta + corr_vector@self.R_inv@zmg
        ''' Sort out dimensions problem '''
        print(y_cap) 
        
        
        
''' mat_file = [filename, space_point_variable_name, param_point_variable_name, output_variable_name]'''        

P = PhIK_1D(mat_file=['test','zu', 'phi0', 'u'])
P.fit()
# P.predict(x_star = 10, q_star = 10)