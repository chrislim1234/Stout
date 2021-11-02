#!/usr/bin/env python
# coding: utf-8

# In[1]:


#There is 1 dataset(csv) with 3 years’ worth of customer orders. There are 4 columns in the csv dataset
#index
#CUSTOMER_EMAIL (unique identifier as hash)
#Net Revenue
#Year


# In[2]:


#Additionally, generate a few unique plots highlighting some information from the dataset. 
#Are there any interesting observations?


# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter


# In[4]:


df_customer_orders = pd.read_csv('customer_orders.csv')
pd.set_option("display.max.columns", None)
df_customer_orders.info()


# In[5]:


df_customer_orders.head(5)


# In[237]:


df2015[['net_revenue']].describe()


# In[79]:


#2015, 2016, 2017
df2015 = df_customer_orders[df_customer_orders['year']==2015]
df2015customers = df2015.customer_email.unique()
df2016 = df_customer_orders[df_customer_orders['year']==2016]
df2016customers = df2016.customer_email.unique()
df2017 = df_customer_orders[df_customer_orders['year']==2017]
print(df_customer_orders.describe())


# In[126]:


#look at customer_email
email2015 = df2015.customer_email
email2016 = df2016.customer_email
email2017 = df2017.customer_email
x = pd.Series(email2015.describe())
y = pd.Series(email2016.describe())
z = pd.Series(email2017.describe())
total = pd.Series(df_customer_orders.customer_email.describe())
index = email2015.describe().index
df = pd.DataFrame(np.array([x,y,z, total]), columns=index)
df.insert(0,
          column='year',
          value=pd.Series(['2015','2016','2017','All years']))
df = df.set_index('year')
print(df)


# In[194]:


#look at customer_email
x = df2015.net_revenue.describe()
y = df2016.net_revenue.describe()
z = df2017.net_revenue.describe()
index = df2017.describe().index
df = pd.DataFrame(np.array([x,y,z]), columns=index)
df.insert(0,
          column='year',
          value=pd.Series([2015,2016,2017]))
df = df.set_index('year')
print(df)
import seaborn
sns.set_palette("Paired", 8, .75)
ax = sns.barplot(x = df.index,
            y = 'count',
            data = df)
ax.set_title('Number of Customers in each Year')
ax.set(xlabel='Year', ylabel='Count')
ax.figure.savefig('Stout/Images/number.png', dpi=300)


# In[134]:


sns.barplot(x = df.index,
            y = 'mean',
            data = df).set_title('Mean of Net Revenue of Customers in each Year')


# In[132]:


sns.barplot(x = df.index,
            y = 'std',
            data = df).set_title('Standard Deviation of Net Revenue of Customers in each Year')


# In[195]:


ax = sns.boxplot(x="year", y="net_revenue", data=df_customer_orders)
ax.set_title('Box plot of Customer Revenue from 2015 to 2017')
ax.set(xlabel='Year', ylabel='Revenue of Customers')
ax.figure.savefig('Stout/Images/boxplot.png', dpi=300)


# In[165]:


x = df2015.net_revenue.sum()
y = df2016.net_revenue.sum()
z = df2017.net_revenue.sum()
df = pd.DataFrame(np.array([x,y,z,x+y+z]), columns=['Total_Revenue'])
df.insert(0,
          column='Year',
          value=pd.Series(['2015','2016','2017','All years']))
df = df.set_index('Year')
ax = sns.barplot(x=df.index, y="Total_Revenue", data=df)


# In[166]:


x = df2015.net_revenue.sum()
y = df2016.net_revenue.sum()
z = df2017.net_revenue.sum()
df = pd.DataFrame(np.array([x,y,z]), columns=['Total_Revenue'])
df.insert(0,
          column='Year',
          value=pd.Series(['2015','2016','2017']))
df = df.set_index('Year')
ax = sns.barplot(x=df.index, y="Total_Revenue", data=df)


# In[224]:


#create sets for each year 
#make each set have a key which is the email and value the revenue for that year
df2017customers = df2017.customer_email.unique()
df2017customerset = set(df2017customers)
df2016customerset = set(df2016customers)
df2015customerset = set(df2015customers)
oldcustomers2015 = df2015customerset.intersection(df2016customerset)
oldcustomers2016 = df2016customerset.intersection(df2017customerset)
# oldcustomerstotal = oldcustomers2015.union(oldcustomers2016)
lostcustomers = df2015customerset.union(df2016customerset).difference(oldcustomerstotal)


# In[196]:


#Total revenue for the current year 2015
customerevenue2015 = df2015.net_revenue.sum()
print("$", customerevenue2015, "total revenue for the current year")


# In[202]:


#Total revenue for the current year 2016
customerevenue2016 = df2016.net_revenue.sum()
print("$", customerevenue2016, "total revenue for the current year")


# In[238]:


#Total revenue for the current year (2017)
customerevenue2017 = df2017.net_revenue.sum()
print("$", customerevenue2017, "total revenue for the current year")


# In[217]:


#New Customer Revenue e.g., new customers not present in previous year only (2016)
newcustomers2016 = df2016customerset.difference(df2015customerset)
newcustomerevenue2016 = df2016[df2016['customer_email'].isin(newcustomers2016)].net_revenue.sum()
print("$", newcustomerevenue2016, "new customer revenue")


# In[241]:


#New Customer Revenue e.g., new customers not present in previous year only (2017)
newcustomers2017 = df2017customerset.difference(df2016customerset)
newcustomerevenue2017 = df2017[df2017['customer_email'].isin(newcustomers2017)].net_revenue.sum()
print("$", newcustomerevenue2017, "new customer revenue")


# In[254]:


#Revenue lost from attrition (2016)
oldcustomerevenue2015 = df2015[df2015['customer_email'].isin(oldcustomers2015)].net_revenue.sum()
oldcustomerevenue2016 = df2016[df2016['customer_email'].isin(oldcustomers2015)].net_revenue.sum() 
x = oldcustomerevenue2015-oldcustomerevenue2016-df2016[df2016['customer_email'].isin(df2016customerset.difference(oldcustomers2015))].net_revenue.sum()
y = df2016[df2016['customer_email'].isin(oldcustomers2016)].net_revenue.sum() 
print(x/y,"revenue attrition")
print("$", x, "revenue lost from attrition", y)
#The formula for revenue attrition is beginning period reoccurring revenue minus end-of-period reoccurring revenue
#divided by beginning period revenue


# In[259]:


#Revenue lost from attrition (2017)
oldcustomers2016 = df2016customerset.intersection(df2017customerset)
oldcustomerevenue2016 = df2016[df2016['customer_email'].isin(oldcustomers2016)].net_revenue.sum()
oldcustomerevenue2017 = df2017[df2017['customer_email'].isin(oldcustomers2016)].net_revenue.sum() 
x = oldcustomerevenue2016-oldcustomerevenue2017-df2017[df2017['customer_email'].isin(df2017customerset.difference(oldcustomers2016))].net_revenue.sum()
y = df2016[df2016['customer_email'].isin(oldcustomers2016)].net_revenue.sum() 
print(x/y,"revenue attrition")
print("$", x, "revenue lost from attrition")
#The formula for revenue attrition is beginning period reoccurring revenue minus end-of-period reoccurring revenue
#divided by beginning period revenue


# In[216]:


#Existing Customer Revenue Current Year (2016)
oldcustomers2016 = df2016customerset.intersection(df2015customerset)
oldcustomerrevenue2016 = df2016[df2016['customer_email'].isin(oldcustomers2016)].net_revenue.sum()
print("$",oldcustomerrevenue2016, "existing customer revenue from current year (2016)")


# In[244]:


#Existing Customer Revenue Prior Year (2016)
customerrevenue2015 = df2015[df2015['customer_email'].isin(oldcustomers2016)].net_revenue.sum()
print(customerrevenue2015, "existing customer revenue from prior year (2015)")


# In[240]:


#Existing Customer Revenue Current Year (2017)
oldcustomers2017 = df2017customerset.intersection(df2016customerset)
oldcustomerrevenue2017 = df2017[df2017['customer_email'].isin(oldcustomers2017)].net_revenue.sum()
print("$",oldcustomerrevenue2017, "existing customer revenue from current year (2016)")


# In[246]:


#Existing Customer Revenue Prior Year (2017)
oldcustomers2017 = df2017customerset.intersection(df2016customerset)
oldcustomerrevenue2016 = df2016[df2016['customer_email'].isin(oldcustomers2017)].net_revenue.sum()
print("$", oldcustomerrevenue2016, "existing customer revenue from prior year (2016)")


# In[223]:


#Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
#2016
print("In 2016, Existing customer growth is ", customerrevenue2016-customerrevenue2015)


# In[248]:


#Existing Customer Growth. To calculate this, use the Revenue of existing customers for current year –(minus) Revenue of existing customers from the previous year
#2017
print("Existing customer growth in 2017 is ", oldcustomerrevenue2017-oldcustomerrevenue2016)


# In[197]:


#Total Customers Current Year (2015)
print(len(df2015customerset), "total customers in current year")


# In[230]:


#Total Customers Current Year (2016)
print(len(df2016customerset), "total customers in current year (2016)")


# In[260]:


#Total Customers Current Year (2017)
print(len(df2017customerset), "total customers in current year (2017)")


# In[231]:


#Total Customers Previous Year (2016)
print(len(df2015customerset), "total customers in previous year (2015)")


# In[261]:


#Total Customers Previous Year (2017)
print(len(df2016customerset), "total customers in previous year (2015)")


# In[233]:


#New Customers (2016)
newcustomers2016 = df2016customerset.difference(df2015customerset)
print(len(newcustomers2016), "new customers (2016)")


# In[262]:


#New Customers (2017)
newcustomers2017 = df2017customerset.difference(df2016customerset)
print(len(newcustomers2017), "new customers (2016)")


# In[235]:


#Old Customers (2016)
oldcustomers2015 = df2015customerset.intersection(df2016customerset)
print(len(oldcustomers2015), "old customers from 2015")


# In[264]:


#Old Customers (2017)
oldcustomers2017 = df2016customerset.intersection(df2017customerset)
print(len(oldcustomers2017), "old customers from 2016")


# In[234]:


#Lost Customers (2016)
oldcustomers2015 = df2015customerset.intersection(df2016customerset)
lostcustomer2016 = df2015customerset.difference(oldcustomers2015)
print(len(lostcustomer2016), "lost customers from 2015")


# In[266]:


#Lost Customers (2017)
oldcustomers2016 = df2016customerset.intersection(df2017customerset)
lostcustomer2017 = df2016customerset.difference(oldcustomers2016)
print(len(lostcustomer2017), "lost customers from 2016")


# In[236]:


#Customer Churn (2016)
oldcustomers2015 = df2015customerset.intersection(df2016customerset)
lostcustomer2016 = df2015customerset.difference(oldcustomers2015)
print(len(lostcustomer2016)/len(df2015customerset), "rate")


# In[267]:


#Customer Churn (2017)
oldcustomers2016 = df2016customerset.intersection(df2017customerset)
lostcustomer2017 = df2016customerset.difference(oldcustomers2016)
print(len(lostcustomer2017)/len(df2015customerset), "rate")


# In[ ]:




