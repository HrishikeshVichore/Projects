import time
import subprocess as sb
import os
import numpy as np
os.chdir(os.getcwd())
times = []
for i in range(10):
    cmd = f'python try1_task1.py'  
    start_time = time.perf_counter()
    sb.run(cmd,shell = True)
    times.append(time.perf_counter() - start_time)
    print(i)
    
avg_time = round(np.mean(times),2)
print(f'{avg_time}\n')
