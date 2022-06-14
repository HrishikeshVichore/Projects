import numpy as np
from sympy import symbols, lambdify, sympify, pprint, Poly, log
import matplotlib.pyplot as plt
from numpy.linalg import inv, cond, lstsq, pinv
from sympy.plotting import plot
from sklearn.preprocessing import normalize

'''y = f(x,a) = x^2 + ax for a from 1 to 5'''

class PhIK():
    def __init__(self, k=None, delta_j=None, f_x=None, space_points=None, param_points=None, h=None, lhc_samples=None):
        np.random.seed(2)
        self.x = symbols('x')
        self.a = symbols('a')
        self.lhc_samples = lhc_samples if lhc_samples else 100
        self.h = [sympify(i) for i in h] if h else [self.x**0, self.x, self.x**2]
        self.space_points = space_points if space_points else list(range(1,11))
        self.param_points = param_points if param_points else list(range(1,6))
        self.f_x = sympify(f_x) if f_x else self.x**2 + self.a*self.x
        self.y = np.array([[float(i.subs(self.x,j)) for j in self.space_points] for i in [self.f_x.subs(self.a,i) for i in self.param_points]])
        
        self.mew = np.mean(self.y, axis = 1) 
        self.k = k if k else lambda y1,mew1,y2,mew2 : np.sum((y1 - mew1)*(y2-mew2))/(self.y.size-1)
        #self.fit()
        # print(self.y)
    def config(self, k=None, delta_j=None, f_x=None, space_points=None, param_points=None, h=None, lhc_samples=None):
        
        if k: self.k = k
        if delta_j: self.delta_j = delta_j
        if space_points: self.space_points = space_points
        if param_points: self.param_points = param_points; self.y = np.array([[float(i.subs(self.space_points,j)) for j in self.space_points] for i in [self.f_x.subs(self.a,i) for i in self.param_points]])
        if f_x: self.f_x = sympify(f_x); self.y = np.array([[float(i.subs(self.x,j)) for j in self.space_points] for i in [self.f_x.subs(self.a,i) for i in self.param_points]])
        if h: self.h = h
        if lhc_samples: self.lhc_samples = lhc_samples
        
    def lambdify_h(self,func):
        return [lambdify(self.x, i, 'numpy') for i in func]
    
    # def cov_func(self, y1,mew1,y2,mew2): 
    #     print(y1,mew1)
    #     print(y2,mew2)
    #     print((y1 - mew1))
    #     print((y2-mew2))
    #     print((y1 - mew1)*(y2-mew2))
    #     print(np.sum((y1 - mew1)*(y2-mew2)))
    #     return np.sum((y1 - mew1)*(y2-mew2))/(self.y.size-1)
    ## for i in range(self.y.shape[0]):
        #     for j in range(self.y.shape[0]):
        #         self.cov_func(self.y[i,:],self.mew[i], self.y[j,:],self.mew[j])
        #         break
        #     break

    
    def least_sq_fit(self,x, y):
        h = self.lambdify_h(self.h)
        A = np.ones((len(self.space_points), len(self.h)))
        for i in range(len(self.h)):
            f = h[i]
            A[:,i] = f(x)
        
        b = lstsq(A,y, rcond = None)[0]
        b[b<=1e-11] = 0
        # b = np.round(b,3)
        
        # print(f'For Space point {x}')
        # print(f'x = {x}')
        # print(f'y(x,a) = {y}')
        # print(f'Basis function = {self.h}')
        # print(f'Matrix of h =\n{A}')
        # print(f'Shape of the matrix = {A.shape}')
        # print('After finding Least Squares Solution')
        # print(f'Beta values =\n{b}')
        # print(f'Shape of the beta vector = {len(b)}')
        
        return b
    
    def fit(self):
        self.h_beta = np.mean([np.multiply(self.least_sq_fit(np.array(self.space_points), self.y[i,:]),self.h) for i in range(self.y.shape[0])])
        # print(self.h_beta)      
        
        self.Inv_Cov_matrix_space = np.array([[self.k(self.y[i,:],self.mew[i], self.y[j,:],self.mew[j]) for j in range(len(self.mew))] for i in range(len(self.mew))])
        # print(self.Inv_Cov_matrix_space)
        # self.Inv_Cov_matrix_space = normalize(self.Inv_Cov_matrix_space, axis = 1, norm = 'l1')
        # print(self.Inv_Cov_matrix_space)
        noise = np.random.normal(np.mean(self.Inv_Cov_matrix_space),np.std(self.Inv_Cov_matrix_space),self.Inv_Cov_matrix_space.shape)
        self.Inv_Cov_matrix_space += noise 
        try:
            self.Inv_Cov_matrix_space = inv(self.Inv_Cov_matrix_space)
        except:
            self.Inv_Cov_matrix_space = pinv(self.Inv_Cov_matrix_space)

        
        # print(self.Inv_Cov_matrix_space)
        print(f'Condition number of Covariance Matrix = {round(cond(self.Inv_Cov_matrix_space),3)}')
        # from PhIK_Try3 import PhIK as try3
        # t = try3()
        # t.fit()
        # self.Inv_Cov_matrix_param = t.Inv_Cov_matrix[0]
        
        # self.R = np.kron(self.Inv_Cov_matrix_space, self.Inv_Cov_matrix_param)
        # print(cond(self.R), self.R.shape)
    
    # def fit1(self):
    #     cov_matrix = []
    #     for i in range(self.y.shape[0]):
    #         for j in range(self.y.shape[1]):
    #             x = self.y[i,j] - self.mew[i]
    #             for k in range(self.y.shape[0]):
    #                 for l in range(self.y.shape[1]):
    #                     y = self.y[k,l] - self.mew[k]
    #                     temp = x*y/(self.y.size-1)
    #                     cov_matrix.append(temp)
    #     cov_matrix = np.array(cov_matrix)
    #     cov_matrix = cov_matrix.reshape(self.y.size,self.y.size)
    #
    #     for i in cov_matrix:
    #         print(i)
    #     print(f'Condition number of Covariance Matrix = {round(cond(cov_matrix),3)}')
    #
    # def fit2(self):
    #
    #     cov_matrix = []
    #     for i in range(self.y.shape[0]):
    #         for j in range(self.y.shape[1]):
    #             x = self.y[i,j] - self.mew[i]
    #             for k in range(self.y.shape[0]):
    #                 for l in range(self.y.shape[1]):
    #                     y = self.y[k,l] - self.mew[k]
    #                     temp = x*y/(self.y.size-1)
    #                     cov_matrix.append(temp)
    #                 break
    #     cov_matrix = np.array(cov_matrix)
    #     cov_matrix = cov_matrix.reshape(len(self.param_points),len(self.space_points),len(self.space_points))
    #
    #     for i in cov_matrix[0]:
    #         print(i)
    #
    #     print(f'Condition number of Covariance Matrix = {round(cond(cov_matrix[1]),3)}')
    #

    def predict(self,q_star,x_star):
        
        h_beta = self.lambdify_h([self.h_beta])[0](x_star)
        y_cap = []
        cov_vector = self.cov_func(q_star,np.array(self.val_a),matrix=False)
        
        for i in range(len(self.X)):
            zero_mean_gaussian = self.val_f_x[:,i] - h_beta #y_data-h_beta
            temp = cov_vector@self.Inv_Cov_matrix[i] #cov_vector @ Inv_Cov_Matrix
            temp = temp@zero_mean_gaussian # Above quantity * zero mean gaussian
            y_cap.append(round(h_beta+temp,2)) # y_cap = h_beta + cov_vector @ Inv_Cov_Matrix @ zero mean gaussian
            
        # print(f'x*={x_star} and q*={q_star}')    
        # print(f'Covariance Vector for q*={q_star} is:-\n{cov_vector}')
        # print(f'h_beta={self.h_beta}\nh_beta evaluated at x*={h_beta}')
        print(f'y_cap for all spatial points =\n{y_cap}')
        return y_cap
    
p = PhIK()
# param_points = list(np.random.uniform(1,5,10))
# p = PhIK(param_points=param_points, f_x='a*x*sin(x)+x**2 * sin(x)', h=['x**0','sin(x)*x**2'])
p.fit()