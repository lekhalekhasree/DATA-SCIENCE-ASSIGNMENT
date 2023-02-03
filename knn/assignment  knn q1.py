#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Prepare a model for glass classification using KNN

Data Description:

RI : refractive index

Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)

Mg: Magnesium

AI: Aluminum

Si: Silicon

K:Potassium

Ca: Calcium

Ba: Barium

Fe: Iron

Type: Type of glass: (class attribute)
        
1 -- building_windows_float_processed
 2 --building_windows_non_float_processed
 3 --vehicle_windows_float_processed
 4 --vehicle_windows_non_float_processed (none in this database)
 5 --containers
 6 --tableware
 7 --headlamps


# In[ ]:


# importing libraries


# In[49]:


# KNN Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# readingf fi;le


# In[50]:


df = pd.read_csv("glass.csv")


# In[51]:


df


# In[ ]:


# eda


# In[52]:


# value count for glass types
df.Type.value_counts()


# In[53]:


df.head()


# In[54]:


df.tail()


# In[55]:


df.info()


# In[56]:


df.describe()


# In[57]:


df.isna().sum()


# In[58]:


#data Visualisation


# In[59]:


#pairwise plot of all the features
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df,hue='Type')
plt.show()


# In[60]:


# Scatter plot of two features, and pairwise plot
sns.scatterplot(df['RI'],df['Na'],hue=df['Type'])


# In[61]:


# correlation matrix 
cor = df.corr()
sns.heatmap(cor)


# In[62]:


df.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));


# In[84]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[85]:


scaler.fit(df.drop('Type',axis=1))


# In[86]:


StandardScaler(copy=True, with_mean=True, with_std=True)


# In[87]:


#perform transformation
scaled_features = scaler.transform(df.drop('Type',axis=1))
scaled_features


# In[88]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[89]:


# APPLYING KNN


# In[91]:


fgmai


# In[92]:


knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')


# In[93]:


knn.fit(X_train,y_train)


# In[94]:


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=4, p=2,
                     weights='uniform')


# In[95]:


y_pred = knn.predict(X_test)


# In[97]:


from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,y_pred))


# In[98]:


accuracy_score(y_test,y_pred)


# In[99]:


k_range = range(1,25)
k_scores = []
error_rate =[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #kscores - accuracy
    scores = cross_val_score(knn,dff,df['Type'],cv=5,scoring='accuracy')
    k_scores.append(scores.mean())
    
    #error rate
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))

#plot k vs accuracy
plt.plot(k_range,k_scores)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

#plot k vs error rate
plt.plot(k_range,error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()


# In[ ]:




