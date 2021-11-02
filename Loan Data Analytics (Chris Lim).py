#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import Counter


# In[4]:


df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None)
df_loans.info()


# In[8]:


nRow, nCol = df_loans.shape
print(f'There are {nRow} rows and {nCol} columns')
print(df_loans.dtypes)


# In[7]:


df_loans.head(5)


# In[9]:


df_loans.dtypes


# In[10]:


df_loans.isnull().sum()


# In[13]:


df_loans.describe(include=np.object)


# In[17]:


df_loans.describe()


# In[16]:


df_loans.describe(include=[np.object,np.number])


# In[8]:


#graph number1 (loan purpose bar graph)
# import seaborn as sns
# count = df_loans['Sector'].value_counts()
# plt.figure(figsize=(15,10))
# ax = sns.countplot(x='Sector', data=df, palette="Set2", order=count.index[0:10])
# ax.set(xlabel='Sectors', ylabel='Number of Companies')
# plt.title("Bar Graph of Sectors")
df_loans.loan_purpose.describe()
loanpurpose = df_loans['loan_purpose'].value_counts()
x = loanpurpose.index
y = loanpurpose.values
#define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:12]
plt.gca().axis("equal")
pie = plt.pie(loanpurpose, startangle=0, autopct='%1.0f%%', pctdistance=0.9, radius=3, colors=colors)
labels=loanpurpose.index.unique()
plt.title('Loan Purpose Pie Chart', weight='bold', size=16, y=2, pad=-20)
plt.legend(pie[0],labels, bbox_to_anchor=(1.5,0.5), loc="center right", fontsize=10, 
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
plt.savefig('Images/pie.png', dpi=300,bbox_inches = 'tight')


# In[111]:


#graph number2 (state data)
df = df_loans['state'].value_counts()
# df2 = pd.DataFrame(np.array([df.index,df.values]), columns=['State','Count'])
# df = df.reset_index()
# df.rename( columns={0:'Count','state':'State'}, inplace=True )
df2 = df.index[0:50]
plt.figure(figsize=(20,10))
ax = sns.countplot(x='state', data=df_loans, palette="Set2", order=df.index[0:10])
ax.set(xlabel='State', ylabel='Count')
plt.title("Bar Graph of States")
plt.savefig('Images/bar.png', dpi=300,bbox_inches = 'tight')


# In[112]:


#graph number3 
# sns.scatterplot(data=df_loans, x="debt_to_income", y="interest_rate")
plt.figure(figsize = (30,16))
ax = sns.displot(data=df_loans, x="interest_rate", hue="application_type", kind="kde", fill=True)
ax.set(xlabel='Interest Rate', ylabel='Density')
plt.title("Interest Rate based on Application type")
plt.savefig('Images/displot.png', dpi=300,bbox_inches = 'tight')


# In[127]:


#graph number3 
# sns.scatterplot(data=df_loans, x="debt_to_income", y="interest_rate")
plt.figure(figsize = (30,16))
ax = sns.boxplot(x="homeownership", y="interest_rate", data=df_loans)
ax.set(xlabel='Homeownership', ylabel='Interest Rate')
plt.title("Interest Rate based on Homeownership", fontsize=14)
plt.savefig('Images/box.png', dpi=300,bbox_inches = 'tight')


# In[126]:


#graph number4 (emp_length to annual_income)
plt.figure(figsize = (30,16))
p = sns.scatterplot(data=df_loans, x="annual_income", y="interest_rate", hue="grade")
p.set(xlabel='Annual Income', ylabel='Interest Rate')
plt.title("Interest Rate based on Annual Income and Corresponding Grade",  fontsize=14)
plt.savefig('Images/scatter.png', dpi=300,bbox_inches = 'tight')


# In[124]:


#graph number5 (compare those with public_record_bankrupt and those without)
df_loans['delinq'] = np.where(df_loans['delinq_2y']== 0, True, False)
# print(df_loans['bankruptcy'])
plt.figure(figsize = (60,60))
p=sns.lmplot(x="debt_to_income_joint", y="interest_rate", col="issue_month", hue="delinq", data=df_loans,
           markers=["o", "x"], palette="Set1");
p.set(xlabel='Debt to Income', ylabel='Interest Rate')
fig = p.fig 
fig.suptitle("Interest Rate based on Debt to Income Ratio", fontsize=12, y=1.08)
plt.savefig('Images/lmplot.png', dpi=300,bbox_inches = 'tight')


# In[290]:


#Create a feature set and create a model which predicts interest rate using at least 2 algorithms. 

#Describe any data cleansing that must be performed and analysis when examining the data.
#Also describe assumptions you made and your approach.
#Visualize the test results and propose enhancements to the model, what would you do if you had more time. 
from sklearn.model_selection import train_test_split
#cleaning and feature set
df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None) 
my_list = df_loans.columns.values.tolist()
# my_list.remove('months_since_90d_late')
# my_list.remove('months_since_last_delinq')
# my_list.remove('debt_to_income_joint')
# my_list.remove('verification_income_joint')
# my_list.remove('annual_income_joint')
# my_list.remove('emp_title')
# my_list.remove('emp_length')
# my_list.remove('months_since_last_credit_inquiry')
# my_list.remove('num_accounts_120d_past_due')
# my_list.remove('state')
objects = df_loans.select_dtypes(include=['object']).columns
for i in objects:
    my_list.remove(i)
df_loans2 = df_loans[my_list].dropna(how='any',axis=0)
df_loans2 = df_loans2.fillna(df_loans2.mean())
y = df_loans2['interest_rate']
my_list.remove('interest_rate')
df2 = df_loans2[my_list]
# df2 = pd.get_dummies(df2, columns=df2.select_dtypes(include=['object']).columns)
# X = df_loans2[my_list]
X = df2
X = X.values #returns a numpy array
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
df = pd.DataFrame(x_scaled)
x = df
# configure to select all features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()
#create dataframe
scores = pd.DataFrame(list(zip(fs.scores_, df2.columns)),
               columns =['FS Score', 'FeatureName'])
scores = scores.sort_values(by=['FS Score'], ascending=False)
scores = scores.set_index('FeatureName')
plt.figure(figsize=(30,10))
# print(scores.FeatureName.values)
x = sns.barplot(x=scores.index, y="FS Score", palette="Set2", data=scores,order=scores.index[0:10])


# In[288]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None) 
y = df_loans['interest_rate']
x = df_loans[['sub_grade','grade','homeownership','loan_purpose']]
x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=12, 
             test_size=0.15)
# with new parameters
gbr = GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3)
# with default parameters
gbr = GradientBoostingRegressor()

gbr.fit(xtrain, ytrain)

ypred = gbr.predict(xtest)
mse = mean_squared_error(ytest,ypred)
print("MSE: %.2f" % mse)

x_ax = range(len(ytest))
plt.figure(figsize=(15,10))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[287]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None) 
y = df_loans['interest_rate']
df = df_loans.drop(df_loans.select_dtypes(include=['int64','float64']).columns, axis=1)
x = df
x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
# print(x)
xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=12, 
             test_size=0.15)
# with new parameters
gbr = GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3)
# with default parameters
gbr = GradientBoostingRegressor()

gbr.fit(xtrain, ytrain)

ypred = gbr.predict(xtest)
mse = mean_squared_error(ytest,ypred)
print("MSE: %.2f" % mse)

x_ax = range(len(ytest))
plt.figure(figsize=(15,10))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


# In[301]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None) 
y = df_loans['interest_rate']
x = df_loans[['paid_interest','public_record_bankrupt','sub_grade','grade']]
x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
xtrain, xtest, ytrain, ytest=train_test_split(x, y, random_state=12, 
             test_size=0.15)
# with new parameters
gbr = GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3)
# with default parameters
gbr = GradientBoostingRegressor()

gbr.fit(xtrain, ytrain)

ypred = gbr.predict(xtest)
mse = mean_squared_error(ytest,ypred)
print("MSE: %.2f" % mse)

x_ax = range(len(ytest))
plt.figure(figsize=(15,10))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
# print(scores.index[0:10])
# paid_interest


# In[330]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
df_loans = pd.read_csv('loans_full_schema.csv')
pd.set_option("display.max.columns", None) 
y = df_loans['interest_rate']
df_loans = df_loans.fillna(df_loans.mean())
x = df_loans.drop('interest_rate',axis=1)
x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
data_dmatrix = xgb.DMatrix(data=x,label=y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100)
xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# In[382]:


from catboost import CatBoostRegressor
my_list = df_loans.columns.values.tolist()
# my_list.remove('months_since_90d_late')
# my_list.remove('months_since_last_delinq')
# my_list.remove('debt_to_income_joint')
my_list.remove('verification_income_joint')
my_list.remove('annual_income_joint')
my_list.remove('emp_title')
# my_list.remove('emp_length')
# my_list.remove('months_since_last_credit_inquiry')
# my_list.remove('num_accounts_120d_past_due')
my_list.remove('state')
# my_list.remove('homeownership')
my_list.remove('verified_income')
# my_list.remove('loan_purpose')
# my_list.remove('application_type')
# my_list.remove('issue_month')
# my_list.remove('loan_status')
# my_list.remove('initial_listing_status')
# my_list.remove('disbursement_method')
objects = df_loans.select_dtypes(include=['object']).columns
df_loans2 = df_loans[my_list].dropna(how='any',axis=0)
df_loans2 = df_loans2.fillna(df_loans2.mean())
y = df_loans2['interest_rate']
my_list.remove('interest_rate')
df2 = df_loans2[my_list]
X = df2
# x = X.values #returns a numpy array
pd.set_option("display.max.columns", None) 
y = df_loans['interest_rate']
# df_loans = df_loans.fillna(df_loans.mean())
# x = df_loans.drop('interest_rate',axis=1)
# x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
CAT_FEATURES = ['grade', 'sub_grade','homeownership','loan_purpose','application_type','issue_month','loan_status','initial_listing_status','disbursement_method'] #list of your categorical features
# set up the model
catboost_model = CatBoostRegressor(n_estimators=50,
                                   loss_function = 'RMSE',
                                   eval_metric = 'RMSE',
                                   cat_features = CAT_FEATURES)
# fit model
catboost_model.fit(X_train, y_train, 
                   eval_set = (X_test, y_test),
                   use_best_model = True,
                   plot = True)
preds = catboost_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
x_ax = range(len(ytest))
plt.figure(figsize=(15,10))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
print("RMSE: %f" % (rmse))


# In[ ]:




