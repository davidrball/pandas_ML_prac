#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:38:34 2020

@author: dball
"""


#going over basics of pandas
import pandas as pd
import numpy as np


#series are 1d arrays, size immutable, value mutable
#can construct just from an array
data =np.array(['a','b','c','d'])
s=pd.Series(data)

#also can pass it a dict
data = {'a':0., 'b':1.,'c':2.}
s=pd.Series(data,index=['b','c','d','a']) #if you pass a dict, sets data corresponding to index, missing value is filled as NaN

#can slice series object just like a list, or retrieve specific indices, e.g.

#print(s[['a','b','c']])

#now onto dataframes, usually has columns of different types, values stored in rows below the columns

#creating dataframe
data = [1,2,3,4,5]
df = pd.DataFrame(data)
#print(df)

#giving it more structured data
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
#print(df)


#can pass it dicts, will automatically assign column keys 
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
#print(df)

#can pass it indices if we want to label each row
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
#print(df)

#can also pass it lists of dicts to fill individual rows, missing vals will be filled with nan
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
#can access specific rows via slices:
dfslice = df[:1] #but can't do individual indices
