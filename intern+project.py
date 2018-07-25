
# coding: utf-8

# In[1]:

#### before loading data inot we need to import required libraries

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
from imblearn.over_sampling import SMOTE


# In[2]:

# reading the data by using the pandas
train=pd.read_csv("C:\\Piazza\\internship\\CSMD\\training.csv")
test=pd.read_csv("C:\\Piazza\\internship\\CSMD\\testing.csv")


# In[3]:

## checking the NULL values in train and test set 
train.isnull().sum()
test.isnull().sum()

## checking the column names in train and test
train.columns
test.columns


# In[4]:

## seeing the summary of data
train.describe()
test.describe

## seeing the first few rows in train and test
train.head(4)
test.head(4)


# In[5]:

#### seeing head of train data
train.head(10)
##### seeing tail of test data
train.tail(10)
print(train.head(5))


# In[6]:

##### seeing head of test data
test.head(10)
#### seeing tail of test data
test.tail(10)
#### both the train and test data show the data frame as each column represented as false


# In[7]:

#### dimension of train data
train.shape
### dimension of test data
test.shape


# In[8]:

#### preprocessing the train and test data
train.isnull().any()
test.isnull().any()


# In[9]:

##### now how many na values in train and test with numeric represent
train.isnull().any()
train.isnull().any(axis=1) #### when axis
##train.isnull.any(axis=1)


# In[10]:

from sklearn import preprocessing
#list(train)
#list(test)
cor_train=train.corr()
cor_test=test.corr()


# In[11]:

plt.matshow(cor_train)
plt.matshow(cor_test)


# In[1]:

#sns.pairplot(train)
#plt.show()


# plot

# In[13]:

sns.heatmap(train.corr(),vmax=1,square=False)
plt.show()


# plotting the different plot

# In[14]:

## frequency of class levels in class variable
train["class"].value_counts()
#test["class"].value_counts()


# In[15]:

## relation between two dates with out class
train.plot(kind="scatter",x="20150330_N",y="20150720_N")
plt.show()
train.head(10)


# In[16]:

sns.jointplot(x="20150330_N",y="20150720_N",data=train,size=5)
plt.show()
list(train.columns)[0]


# for outliers columns and target variable storing into one variable

# In[17]:

#train_drop = train.drop(train.columns[[0,11,16,22,24,25]], axis=1, inplace=True)


# In[18]:

#print(train_drop)


# In[17]:

## boxplot of all variables 
a=sns.boxplot(data=train)
plt.show()


# In[18]:

train['max_ndvi'].dtypes
#type(upper_quartile)


# In[19]:

## checking the whisker value and equal to IQR 
upper_quartile = np.percentile(train['max_ndvi'], 75)
lower_quartile = np.percentile(train['max_ndvi'], 25)
iqr = upper_quartile-lower_quartile
whisker = train['max_ndvi'][train['max_ndvi']>=lower_quartile-1.5*iqr].min()
print(whisker)


# In[20]:


#for i in train['max_ndvi']:
 #   if i <= whisker:
  #      i = whisker
#train.max_ndvi.describe()

train.max_ndvi = train['max_ndvi'].clip(lower = whisker)
sns.boxplot(train.max_ndvi)
plt.show()


# In[21]:

#train['20150210_N']


# In[22]:

train.columns


# In[23]:


train['20150109_N'].dtypes
type(upper_quartile)


# In[24]:

upper_quartile = np.percentile(train['20150109_N'], 75)
lower_quartile = np.percentile(train['20150109_N'], 25)
iqr = upper_quartile-lower_quartile
whisker = train['20150109_N'][train['20150109_N']<=upper_quartile+1.5*iqr].max()
print(whisker)


# In[25]:

train.max_ndvi = train['20150109_N'].clip(upper = whisker)
#sns.boxplot(train.201)
#plt.show()


# In[26]:

upper_quartile_1 = np.percentile(train['20150210_N'], 75)
lower_quartile_1 = np.percentile(train['20150210_N'], 25)
iqr_1 = upper_quartile_1-lower_quartile_1
whisker_z=upper_quartile_1+(1.5*iqr_1)
whisker_1 = train['20150210_N'][train['20150210_N']<=upper_quartile_1+1.5*iqr_1].max()
print(whisker_1,whisker_z)


# In[27]:

for i in train['20150210_N']:
    if i >= whisker_1:
        i = whisker_1
#train['20150210_N'].describe()

#train['20150210_N'] = train['20150210_N'].clip(upper = whisker_1)
sns.boxplot(train['20150210_N'])
plt.show()


# In[28]:



#train['20150210_N'] = train['20150210_N'].clip(upper = whisker2)
#sns.boxplot(train['20150210_N'])
#plt.show()


# In[29]:

sns.boxplot(data=train['20150210_N'])
plt.show()


# In[30]:

q75, q25 = np.percentile(train['20150125_N'], [75, 25])
iqr = q75-q25
whisker3 = q75+(1.5*iqr)
print(whisker3)


# In[31]:

#for i in train['20150125_N']:
 #   if i > whisker3:
  #      i = whisker3


#train['20150125_N'] = train['20150125_N'].clip(upper = whisker3)
sns.boxplot(train['20150125_N'])
plt.show()


# In[32]:

for i in train['20150210_N']:
    if i > whisker:
        i = whisker


train['20150210_N'] = train['20150210_N'].clip(upper = whisker)
sns.boxplot(train['20150210_N'])
plt.show()


# In[ ]:




# In[33]:

q75, q25 = np.percentile(train['20140930_N'], [75, 25])
iqr = q75-q25
whisker = q75+(1.5*iqr)
print(whisker)
for i in train['20140930_N']:
    if i >= whisker:
        i = whisker


train['20140930_N'] = train['20140930_N'].clip(upper = whisker)
sns.boxplot(train['20140930_N'])
plt.show()


# In[34]:

q75, q25 = np.percentile(train["20140423_N"], [75, 25])
iqr = q75-q25
whisker = q75+(1.5*iqr)
print(whisker)
for i in train["20140423_N"]:
    if i >= whisker:
        i = whisker


train["20140423_N"] = train["20140423_N"].clip(upper = whisker)
sns.boxplot(train["20140423_N"])
plt.show()


# In[35]:

upper_quartile_y = np.percentile(train['20140407_N'], 75)
lower_quartile_y = np.percentile(train['20140407_N'], 25)
iqr_y = upper_quartile_y-lower_quartile_y
whisker_y = train['20140407_N'][train['20140407_N']<=upper_quartile_y+1.5*iqr_y].max()
whisker_y1 = upper_quartile_y+(1.5*iqr_y)
print(whisker_y,whisker_y1)



# In[36]:

train['20140407_N'] = train['20140407_N'].clip(upper = whisker_y)
sns.boxplot(train['20140407_N'])
plt.show()


# In[ ]:




# In[37]:

upper_quartile_y = np.percentile(train['20140218_N'], 75)
lower_quartile_y = np.percentile(train['20140218_N'], 25)
iqr_y = upper_quartile_y-lower_quartile_y
whisker_y = train['20140218_N'][train['20140218_N']<=upper_quartile_y+1.5*iqr_y].max()
whisker_y1 = upper_quartile_y+(1.5*iqr_y)
print(whisker_y,whisker_y1)


# In[38]:

#for i in train['20140218_N']:
 #   if i <= whisker:
  #          i = whisker
   #         train['20140218_N'].describe()

train['20140218_N'] = train['20140218_N'].clip(upper = whisker_y)
sns.boxplot(train['20140218_N'])
plt.show()


# In[39]:

upper_quartile_q = np.percentile(train['20140202_N'], 75)
lower_quartile_q = np.percentile(train['20140202_N'], 25)
iqr_q = upper_quartile_q-lower_quartile_q
whisker_q = train['20140202_N'][train['20140202_N']>=lower_quartile_q-1.5*iqr_q].min()
whisker_w = lower_quartile_q-(1.5*iqr_q)
print(whisker_q,whisker_w)


# In[40]:

#for i in train['20140202_N']:
 #   if i <= whisker_q:
  #         i = whisker_q
train['20140202_N'] = train['20140202_N'].clip(lower = whisker_q)


# In[41]:

sns.boxplot(train['20140202_N'])
plt.show()


# In[42]:

sns.boxplot(data = train)
plt.show()


# In[43]:

train.max_ndvi.describe()


# In[45]:

col = list(train.columns)[0]
train.drop(col, axis=1, inplace=True)


# In[39]:

#train.drop(train.columns[16], axis=1, inplace=True)


# In[46]:

from imblearn.over_sampling import RandomOverSampler


# In[47]:

sam = RandomOverSampler(random_state=45)


# In[48]:

X_train = pd.DataFrame (train.iloc[:,1:])
Y_train = pd.DataFrame (train.iloc[:,0])


# In[49]:

xresample,yresample = sam.fit_sample(X_train,Y_train)


# In[50]:

xresample = pd.DataFrame(xresample,columns=X_train.columns)


# In[51]:

yresample = pd.DataFrame(yresample)


# In[52]:

yresample.columns = ['c']


# In[53]:

yresample.c = yresample.c.astype('category')


# In[54]:

yresample.c.value_counts()


# In[ ]:




# In[55]:

yresample.count()


# In[ ]:




# In[2]:

#sns.violinplot(x='class',y="max_ndvi",data=train,size=5)
#plt.show()


# In[59]:

train.shape
test.shape


# In[60]:

train.columns
test.columns
train.shape


# In[61]:

from sklearn import preprocessing
import numpy as np
class_col = train.drop('class', 1)
train_scaled = preprocessing.scale(class_col)
train_scaled.mean(axis=0)
train_scaled.std(axis=0)
class_col2 = test.drop('class',1)
test_scaled = preprocessing.scale(class_col2)
test_scaled.mean(axis=0)
test_scaled.std(axis=0)


# In[21]:

train1 = pd.DataFrame(train_scaled)
test1=pd.DataFrame(test_scaled)
train1.shape
train_class = train['class']
train_data= pd.DataFrame(pd.concat([train_class,train1],axis=1))
train_data = pd.DataFrame(train_data)
train_data.head
test_class = test['class']
test_data = pd.DataFrame(pd.concat([test_class,test1],axis=1))
test_data. head
train_data.head


# In[22]:

train_data.shape
test_data.shape


# Building the models 

# In[24]:

sns.countplot(y="class",data= train)
plt.show()


# In[29]:

#sns.countplot(x="max_ndvi",data=train)
#plt.show()


# In[24]:

sns.heatmap(train_data.corr())
plt.show()
train_data.head
train_data.columns
test_data.columns


# In[26]:

from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression()
#X_train = train_data.iloc[:,0:27]
#train_data['class'] = train_data['class'].astype('category')
#Y_train = train_data.loc[:,'class']
#logreg(X_train,Y_train)


# In[37]:

classifier = LogisticRegression(random_state=0)
X_train = train_data.drop('class',1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[124]:

#### i need to combined the train and test data 
CSMD = pd.concat([train,test])
CSMD.head()
CSMD.describe()
CSMD.shape


# In[125]:

##### to seee the how many positive values and negative values in CSMD
count = CSMD.loc[:,CSMD.columns!='class']
count_p= count[count[:] > 0]
#print(count_p)
count.shape


# In[126]:

count_p.isnull().sum()
sum(count_p.isnull().sum())


# In[127]:

#### now imputing the na values
impute=count_p.fillna(count_p.mean())
impute.isnull().sum()
sum(impute.isnull().sum())


# In[128]:

impute.shape


# In[129]:

CSMD1=pd.concat((impute,CSMD['class']), axis=1)
CSMD1.shape


# In[ ]:




# In[151]:

##### spliting of data into train, validation, test
import sklearn.cross_validation
train1, test1 = sklearn.cross_validation.train_test_split(CSMD1, train_size = 0.7)
train1.shape
test.shape
train_class.shape


# In[143]:

from sklearn import preprocessing
minmaxscaler = preprocessing.MinMaxScaler()
train_class = train1.loc[:,train1.columns!='class']
train_class=train_class.drop_duplicates(inplace=True)
scaled = minmaxscaler.fit_transform(train_class)
data = pd.DataFrame(scaled)
trian_data = pd.concat((data,train1['class']),axis=1)
train_data.shape


# In[144]:

#### preprocessing the data 
from sklearn import preprocessing
import numpy as np
x_train =train1.drop(['class'], axis=1).values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x_train)
x_scaled.shape
x_scaled = pd.DataFrame(x_scaled)
train2 = pd.concat((x_scaled,train1['class']),axis=1)
train1.shape


# In[145]:

### preprocessing the  train1 data 
import numpy as np
from sklearn.preprocessing import StandardScaler
train_class = train1.loc[:,train1.columns!='class']
standardize_train1 = StandardScaler().fit_transform(train_class)
standardize_train = pd.DataFrame(standardize_train1)
standardize_train1.shape
train1.shape
train2 =pd.concat((standardize_train, train1['class']), axis=1)
#test_class = test1.loc[:,test1.columns!=]


# In[72]:

train1.shape
standardize_train.head()


# In[ ]:




# In[30]:

##data standardize 
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

del_col = train.drop('class', 1)
std = preprocessing.StandardScaler()
std.fit(del_col)
x_train = std.transform(del_col)
pd.concat(x_train(1),del_col)


# In[77]:

#### for standardize the data we are using scale to train
del_col = train.drop('class', 1)
scale = preprocessing.scale(del_col)

print(scale_train)
#scale.head(2)
pd.concat(del_col,scale)


# In[76]:

#### for standardize the data on test 
del_col1 = test.drop('class',1)
scaler = preprocessing.scale(del_col1)
print(scaler)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[20]:

np.mean(train)
np.mean(test)


# In[21]:

np.std(train)


# In[31]:

#### for one particular column mean and std 
x=np.mean(train)[2] ###4777.434
y=np.std(train)[2] ###2735.114
print(x-y)


# In[45]:

del_col.head(5)


# In[ ]:




# In[54]:

X_train = pd.DataFrame(train.iloc[:,1:])
Y_train = pd.DataFrame(train.iloc[:,0])
X_test = pd.DataFrame(train.iloc[:,1:])
Y_test = pd.DataFrame(train.iloc[:,0])


# In[55]:

from sklearn import svm 


# In[56]:

svm = svm.SVC(kernel='linear')


# In[ ]:

svm.fit(X_train, Y_train)


# In[ ]:

pred_svm = svm.predict(X_test)

