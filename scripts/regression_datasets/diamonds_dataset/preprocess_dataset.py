import pandas as pd
from sklearn.preprocessing import StandardScaler

'''
This script preprocesses the diamonds dataset and saves it in csv form.
'''

#Load the data
full = pd.read_csv('raw/diamonds.csv', index_col=0)

#Shuffle the data
full = full.sample(frac=1).reset_index(drop=True)

#Change all non-numerix columns to integers
full['cut'] = pd.factorize(full['cut'])[0]
full['color'] = pd.factorize(full['color'])[0]
full['clarity'] = pd.factorize(full['clarity'])[0]

#Scale all features apart from price using a standard normal scaler. Divide the price column by it's maximum value so
# that price has values in [0,1]
scaler = StandardScaler()
full.loc[:,full.columns != 'price'] = scaler.fit_transform(full.loc[:,full.columns != 'price'])
full.loc[:,'price'] = full.loc[:,'price']/full.loc[:,'price'].max()

#Save to folder
full.to_csv('dataset/data.csv')


end =1