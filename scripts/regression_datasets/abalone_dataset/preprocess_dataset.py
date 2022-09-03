import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
This script preprocesses the abalone dataset and saves it in csv form.
'''

#Load the data
full = pd.read_csv('raw/abalone.csv')

#Shuffle the data
full = full.sample(frac=1).reset_index(drop=True)

#Change all non-numerix columns to integers
full['Sex'] = pd.factorize(full['Sex'])[0]

#Scale all features apart from Rings using a standard normal scaler. Divide the Rings column by it's maximum value so
# that price has values in [0,1]
scaler = StandardScaler()
full.loc[:,full.columns != 'Rings'] = scaler.fit_transform(full.loc[:,full.columns != 'Rings'])
full.loc[:,'Rings'] = full.loc[:,'Rings']/full.loc[:,'Rings'].max()

#Save to folder
full.to_csv('dataset/data.csv')



end =1