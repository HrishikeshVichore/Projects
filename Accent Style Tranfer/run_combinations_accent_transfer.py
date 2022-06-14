from subprocess import Popen, PIPE, STDOUT
import os
from itertools import product as pro
from random import choices, seed

seed(4)

style_dir = 'D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Audio_Data_Wav/us_english/'
content = 'D:/Documents/Liclipse Workspace/Accent_Style_Transfer_Project_version_1/Recording.wav'

content_weights = [int(i) for i in [1e1,1e2,1e3]]
style_weights = [1,2,3]
lrs = [0.001,0.002,0.01]
styles = [style_dir + i for i in choices(os.listdir(style_dir), k=3)]

required_lists = [content_weights,style_weights, lrs, styles]
required_combos = list(pro(*required_lists))
print(f'Total Combos : {len(required_combos)}')

params_file = 'params.tsv'
if not os.path.isfile(params_file):
    file = open(params_file, 'a')
    file.write('content\tcontent_loss\tstyle_loss\ttotal_loss\tstyle\tparams\ttime_taken\n')
    file.close()
counter = 1
with open(params_file, 'a') as file:
    for content_weight, style_weight, lr, style in required_combos:
        file_name = os.path.splitext(os.path.basename(content))[0]
        file_name = f'{file_name}_{style_weight}_{content_weight}_{lr}'
        file_name = f'output/audio/{file_name}.wav'
        if os.path.isfile(file_name):
            print(f'Skipping file already exists {file_name}')
            continue
        
        cmd = ['python', 'torch_train.py', "-content", content, "-style", style, "-content_weight", str(content_weight), "-style_weight", str(style_weight), '-learning_rate', str(lr)]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        p.wait()
        out = p.stdout.readlines()[-1]
        out = out.decode('utf-8')
        out = out.split(' ')[2:]
        time_taken = f'{out[0]} {out[1]}'
        content_loss = out[2].split(':')[1]
        style_loss = out[3].split(':')[1]
        total_loss = out[4].split(':')[1].strip()
        params = {'content_weight':content_weight, 'style_weight':style_weight, 'lr':lr}
        line = f'{content},{content_loss},{style_loss},{total_loss},{style},{params},{time_taken}\n'
        file.write(line)
        print(counter)
        #print(line)
        counter += 1
        #break
print('Ran all Combinations and saved result successfully!')        
os.system('shutdown -s -t 60')    
