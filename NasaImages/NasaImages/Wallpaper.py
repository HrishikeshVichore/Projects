import urllib.request as urlr
import os
if os.path.isdir("E:\\Nasa Images"):
    os.chdir("E:\\Nasa Images")
else:
    os.makedirs("E:\\Nasa Images")
    os.chdir("E:\\Nasa Images")

url = 'https://apod.nasa.gov/apod/'
file = open('Nasa.txt','w')
file.write(urlr.urlopen(url).read().decode('utf-8'))
file.close()
metalink = ''
with open("Nasa.txt",'r') as source:
    for line in source:
        if "IMG SRC" in line:
            metalink = line
os.remove("Nasa.txt")
metalink = metalink[10:-2]
url = url + metalink

import random as r

name = "nasa" + str(r.randint(0,100000)) +".jpg"
urlr.urlretrieve(url, name)

import ctypes
ctypes.windll.user32.SystemParametersInfoW(20, 0, "E:\\Nasa Images\\" + name , 0)
