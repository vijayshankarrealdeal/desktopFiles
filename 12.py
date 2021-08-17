import pandas as pd
import numpy as np


df  = pd.read_csv('train.csv')
##toDrop Alley,PoolQC,Fence,MiscFeature

df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis = 1,inplace = True)


val = ['OverallQual','OverallCond','YearBuilt','LotArea','SaleCondition','GarageCars','GarageArea','GarageQual','MSSubClass','LotArea','Neighborhood']

lk = df[val]

lk.isnull().sum()
lk.GarageQual = lk.GarageQual.fillna('TA')

main_data = lk


convert = ['SaleCondition','GarageQual','Neighborhood']
for value in convert:
one_hot = pd.get_dummies(value)
    lk.drop(value,axis = 1,inplace = True)
    lk = df.join(one_hot)