import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Specify the folder path and the Excel file name pattern
folder_path = os.getcwd()+'/Data'
file_name_pattern = 'Ci_Fault_'

for filename in os.listdir(folder_path):
    if filename.startswith(file_name_pattern):
        file_path=os.path.join(folder_path,filename)


        df=pd.read_excel(file_path)
        df['Flag']=0
        # Set values in 'new_column' based on the pattern
        pattern_length = ((((np.average(df['PulseCi'])).astype('int'))/100)*np.average(df['PeriodCi']).astype('int')).astype('int')
        cycle_length = np.average(df['PeriodCi']).astype('int')
    


        for i in range(200, len(df), cycle_length):
            df.loc[i:i + pattern_length - 1, 'new_column'] = 1

        # Print the DataFrame to see the changes
        df['new_column']=df['new_column'].fillna(0)
        df.to_excel(file_path) 



masterfile=pd.DataFrame()
def read_excel_append(folder_path,file_name_pattern,excel_file_path='output.xlsx'):
    try:
        df=pd.read_excel('excel_file_path')
    except FileNotFoundError:
        df=pd.DataFrame()

files=[f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]
for filename in files:
    if filename.startswith(file_name_pattern):
        fileread=os.path.join(folder_path,filename)
        excel=pd.read_excel(fileread)
        excel['tag']=file_name_pattern
        masterfile=masterfile.append(excel)
 
masterfile=masterfile[['tag','SystemResponse_ 4','SystemResponse_ 5','SystemResponse_ 6','SystemResponse_ 7','SystemResponse_ 8','SystemResponse_ 9','SystemResponse_10','new_column']]
masterfile=masterfile.rename(columns={'SystemResponse_ 4':'Ci','SystemResponse_ 5':'Ti','SystemResponse_ 6':'Tci','SystemResponse_ 7':'Tsp','SystemResponse_ 8':'Qc','SystemResponse_ 9':'Tc','SystemResponse_10':'T','new_column':'Flag'})
master=masterfile.to_excel('final1.xlsx')


