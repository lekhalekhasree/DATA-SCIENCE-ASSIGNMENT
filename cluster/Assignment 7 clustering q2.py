#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.

Data Description:
Murder -- Muder rates in different places of United States
Assualt- Assualt rate in different places of United States
UrbanPop - urban population in different places of United States
Rape - Rape rate in different places of United States


# In[ ]:


#importing libraries


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy


# In[ ]:


# reading file


# In[2]:


data=pd.read_csv("crime_data.csv")
data


# In[4]:


data1=data.drop(['Unnamed: 0'],axis=1)
data1


# In[5]:


norm=MinMaxScaler()
df=norm.fit_transform(data1)
df


# In[6]:


df1=pd.DataFrame(df)
df1


# In[23]:


z=linkage(df1,method='average',metric='euclidean')
plt.figure(figsize=(20,9))
plt.title('avg.Dendogram')
plt.xlabel('Index')
plt.ylabel('Dist')
sch.dendrogram(z)
plt.show()


# In[24]:


h=AgglomerativeClustering(n_clusters=4,linkage='average',affinity='euclidean').fit(df1)
h.labels_
labels=pd.DataFrame(h.labels_)
data1['clust']=labels
data1


# In[25]:


data1.groupby(data1.clust).mean()


# In[26]:


data1.clust.value_counts()


# ## KMeans

# In[29]:


plt.figure(figsize=(20,9))
WCSS=[]
for i in range (1,11):
    clf= KMeans(n_clusters=i)
    clf.fit(df1)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show() 


# In[30]:


k=KMeans(n_clusters=4)
pred_k=k.fit_predict(df1)
pred_k


# In[31]:


k.inertia_


# In[32]:


k.cluster_centers_


# In[33]:


m=pd.DataFrame(pred_k)
m


# In[35]:


data2=data.iloc[:,1:]
data2


# In[37]:


data2['clust']=m
data2


# In[38]:


data2.groupby(data2.clust).mean()


# In[39]:


data2.clust.value_counts()


# In[41]:


data0=data.rename({'Unnamed: 0':'State'},axis=1)
data0


# In[42]:


data_k=data0.copy()
data_k


# In[43]:


data_k['clusters']=m
data_k


# In[51]:


plt.figure(figsize=(15,7))
data_k.plot(x="clusters",y ="State",kind="scatter",c=clf.labels_,cmap=plt.cm.Accent)


# ## DBScan

# In[54]:


new=data0.iloc[:,1:]
new


# In[55]:


db=StandardScaler().fit(new)
std=db.transform(new)
std


# In[74]:


dbs=DBSCAN(eps=.8,min_samples=5)
dbs.fit(std)


# In[75]:


dbs.labels_


# In[77]:


new['clust']=dbs.labels_
new


# In[78]:


new.clust.value_counts()


# In[79]:


new


# In[85]:


data_d=pd.concat([data0,new['clust']],axis=1)
data_d


# In[86]:


data_d.plot(x="clust",y ="State",c=dbs.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan') 

