# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing required packages to read and manipulate deta

import pandas as pd
import numpy as np 

#changing display options to see entire output
pd. set_option('display.max_columns', None)
pd. set_option('display.max_rows', None)

#loading detaset
books=pd.read_csv("D:/DataScience/Class/assignment working/Association rule/book.csv")

books.describe()
#removing empty transactions 

#replacing zeros with NA so that we can remove rows with empty transaction
book_1=books.replace(0,np.NaN)
#ckecking Na values
book_1.isna().sum()

#droping transaction full of NA
book_1.dropna(how="all",inplace=True)

#retriving original Zeros
book_1.replace(np.NaN,0,inplace=True)

#importing package to apply association rules
from mlxtend.frequent_patterns import apriori,association_rules

#applying apriori algorithm to calculate  frequent items
frequent_items=apriori(book_1, min_support=0.05, use_colnames=True, max_len=5)
frequent_items.sort_values("support",ascending=False,inplace=True,ignore_index=True)
frequent_items

#visualizing frequent items
import matplotlib.pyplot as plt
plt.bar(x = list(range(0, 11)), height = frequent_items.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_items.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

#applying association rules to form new rules on detaset
rules=association_rules(frequent_items,metric="lift",min_threshold=1)
#reading top 5 rules
rules.head()
#sorting rules by descending order and printing top 10 rules formed
rules.sort_values("lift",ascending=False,ignore_index=True).head(10)
