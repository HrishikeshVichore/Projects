import pandas as pd 
import re
import MySQLdb as ms
FileName    = "g:/SEM-2 2017.csv" 
df          = pd.read_csv(FileName)
df          = df.query("Branch == 'IT'")
df          = df.fillna(-2)
a = df.columns.values
P_F_Columns = []
INT_Columns = []
Sub_Columns = []
T_Columns   = []

a = a[5:]
for i in range(5):
    Sub_Columns.append(a[5*i])
    P_F_Columns.append(a[5*i+1])
    INT_Columns.append(a[5*i+2])
    T_Columns.append(a[5*i+4])
print(*P_F_Columns)
print(*Sub_Columns)
print(*INT_Columns)
print(*T_Columns)    

"""
INT_Columns = []
P_F_Columns = []
Sub_Columns = []
T_Columns   = []
ColVal  =  df.columns.values
for col in ColVal:
    if "INT" in ColVal:
        INT_Columns.append(col)
    elif "R" in ColVal:
        P_F_Columns.append(col)
    elif "T" in ColVal:
        pass
        

P_F_Columns = ['R_AM', 'R_AP', 'R_AC', 'R_ED', 'R_SPA', 'R_CS']
INT_Columns = ['INT_AM', 'INT_AP', 'INT_AC', 'INT_ED', 'INT_SPA', 'INT_CS']
Sub_Columns = ['AM', 'AP', 'AC', 'ED', 'SPA', 'CS']
T_Columns   = ['T_AM', 'T_AP', 'T_AC', 'T_ED', 'T_SPA', 'T_CS']
pd.options.mode.chained_assignment = None
for col in P_F_Columns:
        df[col].replace('P', 1, inplace = True)
        df[col].replace('F', -1, inplace = True)        
        for col in T_Columns:
            df[col].replace('#VALUE!', -5, inplace = True)
for col in Sub_Columns :
    df[col].replace("AB", -3, inplace = True)
con = ms.connect('localhost','root','','miniproject')
Cursor = con.cursor()
for i in range(len(T_Columns)):
    sql  = "SELECT Faculty FROM subject_faculty WHERE Subject = %s "
    data = [T_Columns[i]]
    Cursor.execute(sql,data)
    TeacherName = Cursor.fetchone()
    TeacherName = TeacherName[0]
    print(TeacherName)

con = ms.connect('localhost','root','','miniproject')
df.to_sql(name = FileName, con = con, flavor = "mysql", if_exists = "replace",index = False)
con.commit()
con.close()
df["External_KT"] = 0
df["Internal_KT"] = 0
for row in range(df.shape[0]):
    counter       = 0
    counter1      = 0 
    for col in range (len(P_F_Columns)):
        TheoryMarks   = int(df.iloc[row][Sub_Columns[col]])
        TotalMarks    = int(df.iloc[row][T_Columns[col]])
        InternalMarks = int(df.iloc[row][INT_Columns[col]])
        if df.iloc[row][P_F_Columns[col]] == -1 :
            if((TheoryMarks >= 32) and (TotalMarks < 40) ):
                counter += 1
                df.Internal_KT.iloc[row] = counter 
                
            if((TheoryMarks < 32) and (TotalMarks < 40)  and (InternalMarks < 8)):
                counter += 1
                t = df.Internal_KT.iloc[row] = counter 
            if (TheoryMarks < 32) :
                counter1 += 1
                t = df.External_KT.iloc[row] = counter1 
                
ExtKT       = []
IntKT       = []
for i in range (6):
    int_df  = df["External_KT"].astype(int)
    int_df1 = df["Internal_KT"].astype(int) 
    ExtKT.append(sum(map(lambda x : x == i+1,int_df)))
    IntKT.append(sum(map(lambda x : x == i+1,int_df1)))
#print(type(ExtKT[0]))
#print(IntKT)
"""