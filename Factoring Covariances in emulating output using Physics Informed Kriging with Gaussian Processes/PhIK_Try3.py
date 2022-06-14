import numpy as np
from sympy import symbols, lambdify, sympify, pprint, Poly, log
import matplotlib.pyplot as plt
from numpy.linalg import inv, cond, lstsq
from sympy.plotting import plot
from scipy.stats.qmc import LatinHypercube as lhc
from scipy.stats import qmc
from sympy.matrices import Matrix
from scipy.optimize import minimize, Bounds
from collections import Counter

'''f(x) = x^2 + ax for a from 1 to 5'''

class PhIK():
    def __init__(self, k=None, delta_j=None, f_x=None, X=None, val_a=None, h=None, lhc_samples=None):
        self.x = symbols('x')
        self.a = symbols('a')
        self.lhc_samples = lhc_samples if lhc_samples else 100
        self.h = [sympify(i) for i in h] if h else [self.x**0, self.x, self.x**2]
        self.X = X if X else list(range(1,11))
        self.val_a = val_a if val_a else list(range(1,6))
        self.f_x = sympify(f_x) if f_x else self.x**2 + self.a*self.x
        self.val_f_x = np.array([[float(i.subs(self.x,j)) for j in self.X] for i in [self.f_x.subs(self.a,i) for i in self.val_a]])
        noise = np.random.normal(0,2,self.val_f_x.shape)
        self.val_f_x += noise 
        self.delta_j = delta_j if delta_j else 1
        alpha_j = 2
        self.k = k if k else lambda x,y : np.exp(-(((np.abs(x-y)** alpha_j)/self.delta_j)))
        #self.fit()
        print('Initialized')
    def config(self, k=None, delta_j=None, f_x=None, X=None, val_a=None, h=None, lhc_samples=None):
        
        if k: self.k = k
        if delta_j: self.delta_j = delta_j
        if X: self.X = X
        if val_a: self.val_a = val_a; self.val_f_x = np.array([[float(i.subs(self.x,j)) for j in self.X] for i in [self.f_x.subs(self.a,i) for i in self.val_a]])
        if f_x: self.f_x = sympify(f_x); self.val_f_x = np.array([[float(i.subs(self.x,j)) for j in self.X] for i in [self.f_x.subs(self.a,i) for i in self.val_a]])
        if h: self.h = h
        if lhc_samples: self.lhc_samples = lhc_samples
        
    def lambdify_h(self,func):
        return [lambdify(self.x, i, 'numpy') for i in func]
        
    def cov_func(self, x, y, matrix = True):   
        # print(self.delta_j)     
        if matrix:
            K = np.zeros((len(x),len(y)))
            for i_idx, i in enumerate(x):
                for j_idx, j in enumerate(y):
                    if i_idx <= j_idx:
                        K[i_idx,j_idx] = self.k(i,j)
                    elif i_idx > j_idx:
                        K[i_idx,j_idx] = K[j_idx,i_idx]
            # print(f'Condition Number of Cov Matrix {cond(K)}')
            # K_inv = inv(K)
            # K_inv[K_inv < 1e-12] = 0
            return K
        else:
            return self.k(x,y)
    
    def lhc_sampling(self):
        X=lhc(1,seed=1).random(self.lhc_samples)
        
        X = qmc.scale(X, l_bounds=[1], u_bounds=[10], reverse=False)
        
        X = [i[0] for i in X]
        
        y=np.array([[float(i.subs(self.x,j)) for j in X] for i in [self.f_x.subs(self.a,i) for i in self.val_a]])

        h_beta = np.mean([np.multiply(self.least_sq_fit(X[i], y[:,i]),self.h) for i in range(y.shape[1])])

        zmg = y-h_beta #zmg -> Zero Mean Gaussian -> (Y-hB)
        # EM Algorithm can be tried like GMM.
        def fun(X,f1,f2):
        
            t1 = round(X[0],3)
            t2 = round(f1(X)[0],3)
            t3 = round(f2(X)[0],3)
            loss = np.sqrt(np.square(np.subtract(t3,t2)).mean())
            return loss
        def callback(X):
            # print(f1)
            # print('X\tf1\tf2')
            t1 = round(X[0],3)
            t2 = round(f1(X)[0],3)
            t3 = round(f2(X)[0],3)
            loss = np.sqrt(np.square(np.subtract(t3,t2)).mean())
            print(f'{t1}\t{t2}\t{t3}\t{loss}')
            
        # print(f'f1 -> {zmg[0][2]}\nf2 -> {-log(zmg[0][2])}')
        # print('X\tf1\tf2\tloss')
        l_values = []
        for i in range(zmg.shape[0]):
            for j in range(zmg.shape[1]):
                f1 = zmg[0][2]
                f2 = -log(f1)
                f1,f2 = self.lambdify_h([f1,f2])
                b = Bounds(0.00001,5)
                # b = None
                PrintParams = minimize(fun,[1.5],args = (f1,f2),method='L-BFGS-B',bounds = b,callback = callback)
                l_values.append(PrintParams['x'][0])
                # print(PrintParams['x'][0])
                break
            break
        # l = set(l_values)
        # print(l)
        # print(len(l))
        # l = Counter(l_values)
        # print(l)
        # # plt.bar(l.keys(), l.values())
        # plt.hist(l_values)
        # plt.xlabel('values of delta_j')
        # plt.ylabel('Count')
        # plt.show()
        #
        l = max(l_values, key=l_values.count)
        
        return l
    
    def least_sq_fit(self,x, y):
        h = self.lambdify_h(self.h)
        A = np.ones((len(self.val_a), len(self.h)))
        for i in range(len(self.h)):
            f = h[i]
            A[:,i] = f(x)
        
        b = lstsq(A,y, rcond = None)[0]
        b[b<=1e-11] = 0
        b = np.round(b,3)
        
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
    
    def fit(self,show_loss=False):
        self.h_beta = np.mean([np.multiply(self.least_sq_fit(self.X[i], self.val_f_x[:,i]),self.h) for i in range(self.val_f_x.shape[1])])
        # self.delta_j = self.lhc_sampling()
        print(self.delta_j)
        self.Inv_Cov_matrix = np.array([self.cov_func(self.val_a, self.val_a, matrix=1) for i in range(len(self.X))])
        # L = np.linalg.cholesky(self.Inv_Cov_matrix)
        # print(L)
        mean_f_x = self.f_x.subs(self.a,3)
        print(f'Condition Number of Cov Matrix {cond(self.Inv_Cov_matrix[0])}')
        if show_loss:
            RMSE = self.loss(mean_f_x,self.h_beta)
            print(f'Loss from true mean is {RMSE}')
        
    def plotit(self):
        for xe, ye in zip(self.val_a, self.val_f_x):
            plt.scatter([xe] * len(ye), ye)
        plt.title(f'f(x,a)={self.f_x}')
        plt.xlabel('values of a')
        plt.ylabel('f(x)')
        plt.xticks(self.val_a)
        #plt.yticks(list(range(5,155,5)))
        #plt.plot(y_cap)
        plt.show()
        mean_f_x = self.f_x.subs(self.a,3)
        plot(self.h_beta, mean_f_x,(self.x,-20,20),legend = 1)
    
    def loss(self,x,y):
        x,y = self.lambdify_h([x,y])
        samples = np.array(range(20))
        RMSE = np.sqrt(np.square(np.subtract(x(samples),y(samples))).mean())
        return RMSE
        
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
    

# np.random.seed(2)
# val_a = list(np.linspace(1.0, 5.0, num=10))
# val_a = list(np.random.uniform(1,5,5))
# P = PhIK(val_a=val_a,f_x='a*x*sin(x)+x**2 * sin(x)', h=['x**0','sin(x)*x**2'],delta_j=None)
P = PhIK()
P.fit()
# y_cap = P.predict(q_star=3.5,x_star=2.5)
# P.plotit()




