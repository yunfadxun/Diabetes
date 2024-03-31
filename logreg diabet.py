#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


df =pd.read_csv("diabetes2.csv")
df


# In[14]:


df.Insulin.min()


# In[17]:


col=df.columns[0:8]
col


# In[29]:


plt.subplot(1,2,1)
plt.boxplot(df.query("Outcome==1")[col].values)
plt.subplot(1,2,2)
plt.boxplot(df.query("Outcome==0")[col].values)


# In[30]:


df.corr()


# In[38]:


sns.heatmap(df.corr(),annot=True)


# In[169]:


#since blood pressure and pedigree appear to be less powerful predictor, drop it
df2=df.drop(columns=["BloodPressure","DiabetesPedigreeFunction"])
df2.info()


# In[84]:


sns.heatmap(df2[df2.columns[0:6]].corr(),annot=True)


# In[85]:


sns.heatmap(df2.corr(),annot=True)


# In[86]:


df.info()


# In[87]:


df2


# In[88]:


df2["Outcome"]=df2["Outcome"].replace({1:"Diabetes",0:"Healthy"})


# In[89]:


df2


# In[102]:


y= df2["Outcome"].value_counts()


# In[101]:


x= df2["Outcome"].unique()


# In[103]:


plt.bar(x,y)


# In[119]:


p= df2[df2.columns[0:6]]


# In[128]:


outcome= df2[["Outcome"]]


# In[143]:


#data splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(p,outcome,random_state=42, train_size=0.8)

print("train size X : ",x_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",x_test.shape)
print("test size y : ",y_test.shape)


# In[145]:


#normalize dataset or convert into z score
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)


# In[146]:


#check distribution
y_train.value_counts(normalize=True)


# In[149]:


#building logreg
#since y is in a form of 1D Array, convert it into column-vector
from sklearn.linear_model import LogisticRegression
basemodel= LogisticRegression()
basemodel.fit(x_train,y_train.values.ravel())


# In[156]:


#data train
y_pred_basemodel= basemodel.predict(x_test)


# In[160]:


#obtain f1 score
#since y is categorical variable, put average=none, otherwise leave it as it is if it's 0/1 (binary variable)
#result is 0.622 predicted for 0=healthy and 0.801 for diabetes
from sklearn.metrics import f1_score
print("f1 score for basemodel is: ", f1_score(y_test, y_pred_basemodel, average=None))


# In[162]:


#create confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred_basemodel)


# In[165]:


#create classification report
from sklearn.metrics import classification_report
classification_report(y_test,y_pred_basemodel).split("\n")


# In[168]:


#create confusion matrix display
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_basemodel)).plot()


# In[ ]:




