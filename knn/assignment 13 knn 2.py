#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Implement a KNN model to classify the animals in to categorie


# In[ ]:


# import libraries


# In[29]:


# KNN Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# reading file


# In[30]:


df = pd.read_csv("Zoo (1).csv")


# In[31]:


df


# In[32]:


from sklearn import preprocessing


# In[33]:


label_encoder = preprocessing.LabelEncoder()
df['animal name']= label_encoder.fit_transform(df['animal name'])
df


# In[ ]:


# eda


# In[34]:


df.describe()


# In[35]:


df.info()


# In[36]:


df.head()


# In[37]:


df.tail()


# In[38]:


df.isna().sum()


# In[39]:


df.type.value_counts()


# In[40]:


#data Visualisation


# In[41]:


df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));


# In[42]:


# correlation matrix 
cor = df.corr()
sns.heatmap(cor)


# In[46]:


# Scatter plot of two features, and pairwise plot
sns.scatterplot(df['animal name'],df['type'])


# In[47]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[49]:


scaler.fit(df.drop('type',axis=1))


# In[50]:


StandardScaler(copy=True, with_mean=True, with_std=True)


# In[52]:


#perform transformation
scaled_features = scaler.transform(df.drop('type',axis=1))
scaled_features


# In[53]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[54]:


# APPLYING KNN


# In[61]:



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
x,y = df.loc[:,df.columns != 'hair'], df.loc[:,'hair']
knn.fit(x,y)
prediction = knn.predict(x)
print("Prediction = ",prediction)


# In[62]:


#Train Test Split


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 1)
x,y = df.loc[:,df.columns != 'hair'], df.loc[:,'hair']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print('With KNN (K=1) accuracy is: ',knn.score(x_test,y_test)) # accuracy


# In[66]:


k_values = np.arange(1,25)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(k_values):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))


# In[67]:


# Plot
plt.figure(figsize=[13,8])
plt.plot(k_values, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_values, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# In[ ]:




