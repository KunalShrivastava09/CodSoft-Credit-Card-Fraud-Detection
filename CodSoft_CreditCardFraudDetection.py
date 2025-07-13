#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np 


# In[3]:


from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report, roc_curve, auc


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv(r"C:\Users\hp\Downloads\creditcard_2023.csv")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


X=df.drop({'id','Class'}, axis=1, errors='ignore')##  errors ignore in case some coulumns don't exist 
y=df['Class']


# In[10]:


print(X.columns.tolist)


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


X_train.shape


# In[13]:


X_test.shape


# In[14]:


scaler= StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# In[15]:


print(pd.Series(Y_train).value_counts(normalize=True))


# In[17]:


rf_model = RandomForestClassifier(
    n_estimators=20, 
    max_depth=10, 
    random_state=42, 
    n_jobs=-1  # use all CPU cores
)



# In[18]:


rf_model.fit(X_train_scaled, Y_train)


# In[19]:


y_pred=rf_model.predict(X_test_scaled)


# In[20]:


print(classification_report(Y_test, y_pred))


# In[21]:


plt.figure(figsize=(8,6))  # ✅ use figsize, not figure
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[22]:


importance=rf_model.feature_importances_
feature_imp=pd.DataFrame({
    'Feature':X.columns,
    'Importance':importance
}).sort_values('Importance',ascending=False)


# In[23]:


feature_imp.head()


# In[24]:


plt.figure(figsize=(10,6))
sns.barplot(data=feature_imp, x='Importance', y='Feature')  # Fix X/Y and capitalization
plt.title('Feature Importance Ranking')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(12,8))
correlation_matrix=X.corr()
sns.heatmap(correlation_matrix,cmap='coolwarm',center=0,annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()


# In[26]:


# Get predicted probabilities for the positive class (1)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)

# Compute AUC
roc_auc = auc(fpr, tpr)
print("AUC Score:", roc_auc)


# In[27]:


plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve(AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.xlim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Trye Positive Rate')
plt.title('Receiver Operating Charecteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




