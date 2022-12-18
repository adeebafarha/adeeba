#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#1
data=pd.read_csv("C:/Users/ACER/Downloads/titanic_dataset.csv")


# In[3]:


data


# In[19]:


#2
data=pd.read_csv("C:/Users/ACER/Downloads/titanic_dataset.csv",index_col="PassengerId")


# In[21]:


data


# In[22]:


#3
data.shape


# In[23]:


data.info()


# In[24]:


#4
data.isna().sum()


# In[26]:


data.columns


# In[40]:


num_cols=data[[ 'Age']]


# In[41]:


num_cols.isna().sum()


# In[30]:


from sklearn.impute import SimpleImputer


# In[67]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') 


# In[68]:


num_cols = imputer.transform(num_cols)
imputer = imputer.fit(num_cols)


# In[49]:





# In[64]:


type(num_cols)


# In[65]:


num_cols=pd.DataFrame(num_cols,columns=['Age'])


# In[52]:


type(num_cols)


# In[53]:


num_cols.isna().sum()


# In[54]:


data=data.drop(['Age'],axis=1)


# In[55]:


data=pd.concat([num_cols,data],axis=1)


# In[62]:


data


# In[77]:


plt.boxplot(data['Parch'])


# In[78]:


plt.title('Boxplot for Parch')


# In[79]:


plt.boxplot(data['Fare'])


# In[80]:


plt.title('Boxplot for Fare')


# In[81]:


plt.boxplot(data['SibSp'])


# In[82]:


plt.title('Boxplot for SibSp ')


# In[ ]:




