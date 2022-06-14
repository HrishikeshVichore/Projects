
import numpy as np
import pandas as pd
import math

def f (w, b, x):
    return 1.0/(1.0 + np.exp(-(w*x+b))) 

def grad_b (w, b, x, y):
    fx = f(w, b, x)
    return(fx - y) * fx * (1 - fx)

def grad_w (w, b, x, y):
    fx = f(w, b, x)
    return(fx - y) * fx * (1 - fx) * x

def error (w, b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err

df = pd.read_csv('validation_data.csv')
data = df.values
X = data[:,0]
Y = data[:,1]

w_b_dw_db = [(1,1,0,0)]
w_history, b_history, error_history = [], [], []
w, b, eta, mini_batch_size, num_points_seen = 1, 1, 0.01, 10, 0
m_w, m_b, v_w, v_b, eps, beta1, beta2 = 0, 0, 0, 0, 1e-8, 0.9, 0.999
for i in range(100):
    dw, db = 0, 0
    for x,y in zip(X,Y):
        dw += grad_w(w, b, x, y)
        db += grad_b(w, b, x, y)
        
    m_w = beta1 * m_w + (1 - beta1)*dw
    m_b = beta1 * m_b + (1 - beta1)*db
    
    v_w = beta2 * v_w + (1 - beta2)*dw**2
    v_b = beta2 * v_b + (1 - beta2)*db**2
    
    m_w = m_w/(1 - math.pow(beta1, i+1))
    m_b = m_b/(1 - math.pow(beta1, i+1))
    
    v_w = v_w/(1 - math.pow(beta2, i+1))
    v_b = v_b/(1 - math.pow(beta2, i+1))
    
    w = w - (eta / np.sqrt(v_w + eps)) * m_w
    b = b - (eta / np.sqrt(v_b + eps)) * m_b

err = error(w, b)
print(err)

"""

import numpy as np

w = np.load('epochs/weights_after_epoch_0.npy')
print(w)

"""