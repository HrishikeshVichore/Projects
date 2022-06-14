import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance
from collections import OrderedDict
from scipy import stats

if __name__ == '__main__':
    Total_Descriptors = []

    dirPath = 'H:/D Drive/test1'
    for i in range(0,1000):
        filePath = dirPath + '/' + str(i) + '.jpg.pickle'
        file = open(filePath, 'rb')
        data = np.asarray(pickle.load(file))
        file.close()
        data = stats.zscore(data)
        Total_Descriptors.append(data)
        print(i)
        #print(data)

    query = int(input('Enter query Image : '))
    result = {}
    for i in range (0, 1000):
        p = distance.euclidean(Total_Descriptors[i], Total_Descriptors[query])
        result[i] = p
    print(result)

    od = OrderedDict(sorted(result.items(), key=lambda x: x[1]))
    for key, value in od.items():
        print(key, ',', value)
