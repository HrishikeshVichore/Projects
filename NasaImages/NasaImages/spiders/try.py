import pickle

p = open('output.pickle','rb')
p = pickle.load(p)
print(p)