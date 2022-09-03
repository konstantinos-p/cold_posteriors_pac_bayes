import pandas as pd
from sklearn.preprocessing import StandardScaler


'''
This script preprocesses the KC_House dataset and saves it in csv form.
'''

#Load the data
full = pd.read_csv('raw/kc_house_data.csv')

#Shuffle the data
full = full.sample(frac=1).reset_index(drop=True)

#Delete columns that are not useful
full = full.drop(['id','date'],axis=1)


#Change all non-numerix columns to integers
full['zipcode'] = pd.factorize(full['zipcode'])[0]


#Scale all features apart from price using a standard normal scaler. Divide the price column by it's maximum value so
# that price has values in [0,1]
scaler = StandardScaler()
full.loc[:,full.columns != 'price' ] = scaler.fit_transform(full.loc[:,full.columns != 'price'])
full.loc[:,'price'] = full.loc[:,'price']/full.loc[:,'price'].max()

#Save to folder
full.to_csv('dataset/data.csv')


end =1