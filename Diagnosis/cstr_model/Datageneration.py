import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_excel('Ci_Fault_15_1.xls')
df['Flag']=0
# Set values in 'new_column' based on the pattern
pattern_lengt = (np.average(df['PulseCi'])).astype('int')/100

cycle_length = np.average(df['PeriodCi']).astype('int')
pattern_length = pattern_lengt*cycle_length
print(pattern_length,cycle_length)


for i in range(200, len(df), cycle_length):
    df.loc[i:i + pattern_length - 1, 'new_column'] = 1

# Print the DataFrame to see the changes
df['new_column']=df['new_column'].fillna(0)


plt.plot(df.index, df['SystemResponse_ 1'],label='ci')
plt.plot(df.index, df['new_column'],label='flag')
plt.xlabel('index')
plt.ylabel('response')



# Add legend
plt.legend()

# Show the plot
plt.show()
