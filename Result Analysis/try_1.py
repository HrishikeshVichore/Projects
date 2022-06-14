import matplotlib.pyplot as plt
import pandas as pd


filename = 'SEM-2 2017.csv'
sep      = ','
P_F_Columns = ['R_AM', 'R_AP', 'R_AC', 'R_ED', 'R_SPA', 'R_CS']
INT_Columns = ['INT_AM', 'INT_AP', 'INT_AC', 'INT_ED', 'INT_SPA', 'INT_CS']
T_Columns   = ['T_AM', 'T_AP', 'T_AC', 'T_ED', 'T_SPA', 'T_CS']
df = pd.read_csv(filename,sep=sep)
df  = df.query("Branch == 'IT'")
df  = df.fillna(-2)
df  = df.set_index('Sr_No')
s=df.shape
s=s[0]-1      #Appeared
if not s == -1:
    dict1={}
    dict2={}
    dict3={}
    dict4={}
    for col in P_F_Columns:
        df[col].replace('P', 1, inplace = True)
        df[col].replace('F', -1, inplace = True)        
    for col in T_Columns:
        df[col].replace('#VALUE!', -5, inplace = True)
    #print(df)
print(df.columns.values)

for row in range(df.shape[0]):
    ekt = 0
    ikt = 0
    for col in range(len(P_F_Columns)):
        if df.iloc[row][P_F_Columns[col]] == -1 :
            A = int(df.iloc[row][INT_Columns[col]])
            B = int(df.iloc[row][T_Columns[col]])
            if A < 8 and B <= 40 :
                ikt += 1        
            else:
                ekt += 1
            #print("Name = ",df.iloc[i][1] , i, ' : ', "Internal KT = ", ikt, "External KT = ", ekt,"Subject = ",P_F_Columns[col])
import csv
reader = csv.DictReader(open(filename))
f =  reader.fieldnames
print(f)


"""
    for col in T_Columns:
        int_df      = df[col].astype(int)
        below_lvl   = map(lambda x: x < 50, int_df)
        below_lvl   = (sum(below_lvl)/s)*100
        between_lvl = map(lambda x: 50 < x < 60, int_df)
        between_lvl = (sum(between_lvl)/s)*100
        above_lvl   = map(lambda x: x > 60, int_df)
        above_lvl   = (sum(above_lvl)/s)*100
        dict1[col]  = below_lvl 
        dict2[col]  = between_lvl
        dict3[col]  = above_lvl  
dict4["Below_lvl"]   = dict1
dict4["Between_lvl"] = dict2
dict4["Above_lvl"]   = dict3     
print(list(dict1.values()))
print(list(dict1.keys()))
print(dict1)
list1=[dict1,dict2,dict3]
for dict in list1:
    left = [1, 2, 3, 4, 5, 6]
    height = list(dict.values())
    tick_label = list(dict.keys())
    plt.bar(left, height, tick_label = tick_label,
            width = 0.8, color = ['red', 'green'])
    plt.xlabel('Subject')
    plt.ylabel('Below 50%')
    plt.title('My bar chart!')
    plt.show()
"""
