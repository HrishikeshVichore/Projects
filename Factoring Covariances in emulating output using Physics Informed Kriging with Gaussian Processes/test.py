from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = pd.read_csv('Data/UT.csv')
number_of_space_points = 51

space_points = file.iloc[:,:number_of_space_points].to_numpy()[0]

file = pd.read_csv('Data/UT_Test.csv')
original = file.iloc[:,2:].to_numpy()

file = pd.read_csv('Data/for_plot_new.csv',header=None)
predicted = file.to_numpy()
print(original.shape)
print(predicted.shape)


for i in range(original.shape[0]):
    plt.plot(space_points, original[i],label='Original')
    plt.plot(space_points, predicted[i],label='Predicted')
    plt.legend()
    plt.show()
