#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Build a simpleMuliple regression model by performing EDA and do necessary transformations and elect the best model using  Python.
# Prepare a prediction model for profit of 50_startups data.Do transformations for getting better predictions of profit and make a table containing R^2 value for each prepared model.


# In[ ]:


#BASIC LIBRARIES


# In[123]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[124]:


df = pd.read_csv("ToyotaCorolla (1).csv",encoding='latin1')
df = df.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)


# In[125]:


df


# In[126]:


df =pd.concat([df.iloc[:,2:4],df.iloc[:,6:7],df.iloc[:,8:9],df.iloc[:,12:14],df.iloc[:,15:18]],axis=1)


# In[127]:


df


# In[ ]:


#EDA


# In[128]:


df.info()


# In[129]:


df.min()


# In[130]:


df.describe()


# In[ ]:


#VISUALIZATION


# In[131]:


sns.pairplot(df)


# In[132]:


df.corr()


# In[133]:


# PREPARING A MODEL


# In[134]:


#Build model
import statsmodels.formula.api as smf 
model = smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df).fit()


# In[135]:


model.summary()


# In[136]:


model.params


# In[137]:


model.tvalues


# In[138]:


model.pvalues


# In[139]:


model.resid


# In[140]:


model.resid_pearson


# In[141]:


#simple linear regression


# In[142]:


ml_cc=smf.ols('Price~CC',data = df).fit()   
#t and p-Values
print(ml_cc.tvalues, '\n', ml_cc.pvalues) 


# In[143]:


ml_doors=smf.ols('Price~Doors',data = df).fit()   
#t and p-Values
print(ml_doors.tvalues, '\n', ml_doors.pvalues) 


# In[144]:


ml_doorscc=smf.ols('Price~Doors+CC',data = df).fit()   
#t and p-Values
print(ml_doorscc.tvalues, '\n', ml_doorscc.pvalues) 


# In[145]:


# Calculating VIF


# In[146]:


rsq_age = smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared  
vif_age = 1/(1-rsq_age) 
rsq_km = smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared  
vif_km = 1/(1-rsq_km) 
rsq_hp = smf.ols('HP~KM+Age+CC+Doors+Gears+QT+Weight',data=df).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 
rsq_cc = smf.ols('CC~KM+HP+Age+Doors+Gears+QT+Weight',data=df).fit().rsquared  
vif_cc = 1/(1-rsq_cc) 
rsq_doors = smf.ols('Doors~KM+HP+CC+Age+Gears+QT+Weight',data=df).fit().rsquared  
vif_doors = 1/(1-rsq_doors)
rsq_gears = smf.ols('Gears~KM+HP+CC+Doors+Age+QT+Weight',data=df).fit().rsquared  
vif_gears = 1/(1-rsq_gears) 
rsq_qt = smf.ols('QT~KM+HP+CC+Doors+Gears+Age+Weight',data=df).fit().rsquared  
vif_qt = 1/(1-rsq_qt) 
rsq_weight = smf.ols('Weight~KM+HP+CC+Doors+Gears+QT+Age',data=df).fit().rsquared  
vif_weight = 1/(1-rsq_weight) 


# In[147]:


# Storing vif values in a data frame
d1 = {'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_doors,vif_gears,vif_qt,vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[148]:


# Residual Analysis


# In[149]:


#Test for Normality of Residuals (Q-Q Plot)


# In[150]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[151]:


list(np.where(model.resid>10)) 


# In[152]:


#Residual Plot for Homoscedasticity


# In[153]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[154]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[155]:


#Residual Vs Regressors


# In[156]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
plt.show()


# In[157]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[158]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[159]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "CC", fig=fig)
plt.show()


# In[160]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Doors", fig=fig)
plt.show()


# In[161]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Gears", fig=fig)
plt.show()


# In[162]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "QT", fig=fig)
plt.show()


# In[163]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# In[164]:


#Model Deletion Diagnostics


# In[165]:


#Detecting Influencers/Outliers


# In[166]:


#Cook’s Distance


# In[167]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[168]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(c, 3))  
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[169]:


(np.argmax(c),np.max(c)) 


# In[170]:


#High Influence points


# In[171]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[172]:


k = 8
n = df.shape[0]
levarege_cutoff = 3*((k+1)/n)
levarege_cutoff


# In[173]:


#From the above plot, it is evident that data point 80 are the influencers¶


# In[174]:


df[df.index.isin([80])] 


# In[175]:


#See the differences in RD and other variable values
df.head()


# In[176]:


#Improving the model¶


# In[177]:


df1=df.drop(df.index[80],axis=0).reset_index()
df1=df1.drop(['index'],axis=1)
df1


# In[178]:


#build model
#. excluding cc


# In[179]:


import statsmodels.formula.api as smf 
final_ml_cc=smf.ols('Price~Age+KM+HP+Doors+Gears+QT+Weight',data = df).fit() 
final_ml_cc


# In[180]:


final_ml_cc.params


# In[181]:


final_ml_cc.resid


# In[182]:


(final_ml_cc.rsquared)


# In[183]:


(final_ml_cc.aic,final_ml_cc.bic)


# In[184]:


#. excluding doors


# In[185]:


final_ml_doors=smf.ols('Price~Age+KM+HP+Gears+QT+Weight+CC',data=df1).fit()
final_ml_doors


# In[186]:


(final_ml_doors.rsquared,final_ml_doors.rsquared_adj)


# In[187]:


(final_ml_doors.tvalues,final_ml_doors.pvalues)


# In[188]:


#Again check for influencers
model_influence_doors = final_ml_doors.get_influence()
(c_doors, _) = model_influence_doors.cooks_distance


# In[189]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.round(c_doors,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[190]:


(np.argmax(c_doors),np.max(c_doors)) 


# In[191]:


df2=df1.drop(df.index[[220]],axis=0) 


# In[192]:


df2


# In[193]:


#Reset the index and re arrange the row values
df3=df2.reset_index()  


# In[194]:


df4=df3.drop(['index'],axis=1)


# In[195]:


df4


# In[196]:


#Check the accuracy of the mode
final_ml_doors= smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=df4).fit()


# In[197]:


final_ml_doors.rsquared


# In[198]:


final_ml_doors.aic


# In[199]:


final_ml_doors.bic


# In[200]:


model_influence_doors = final_ml_doors.get_influence()
(c_doors, _) = model_influence_doors.cooks_distance


# In[201]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.round(c_doors,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[202]:


(np.argmax(c_doors),np.max(c_doors))


# In[203]:


new =df4.copy()
new


# In[204]:


df4[df4.index.isin([958])]


# In[205]:


df5 =new.drop(new.index[[958]],axis=0).reset_index(drop=True)


# In[206]:


df5


# In[207]:


df5[df5.index.isin([958])]


# In[208]:


model3=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+Weight',data=df5).fit()
model3.summary()


# In[209]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.round(c_doors, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[210]:


np.argmax(c_doors),np.max(c_doors)


# In[211]:


new2 =df5.copy()
new2


# In[212]:


df5[df5.index.isin([958])]


# In[213]:


df6=new2.drop(new2.index[[958]],axis=0).reset_index(drop=True)
df6


# In[214]:


df6[df6.index.isin([958])]


# In[215]:


model3=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+Weight',data=df6).fit()
model3.summary()


# In[216]:


model_influence3 = model3.get_influence()
(c_doors, _) = model_influence3.cooks_distance
c_doors


# In[217]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.round(c_doors, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[218]:


np.argmax(c_doors),np.max(c_doors)


# In[219]:


new3=df6.copy()
new3


# In[220]:


df7 =new3.drop(new3.index[[599]],axis=0).reset_index(drop=True)
df7


# In[226]:


# say New data for prediction is
new_data=pd.DataFrame({'Age':34,"KM":47000,"HP":90,"CC":1350,"Doors":4,"Gears":6,"Tax":69,"Weight":1017},index=[0])
new_data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




