import pandas as pd

df = pd.read_csv('remaining_combinations.csv')
a = df['features']
b = df['epochs']
c = df['batch_size']
d = df['optimizer']
e = df['loss']
f = df['activation']
g = df['metrics']
count = 0
for fe,epochs,batch_size,optimizer,activation,metrics in zip(a,b,c,d,f,g):
    print(fe,type(epochs),type(batch_size),type(optimizer),type(activation),type(metrics))
    
    count += 1
print(count)