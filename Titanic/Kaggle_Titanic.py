#!/usr/bin/env python
# coding: utf-8

# In[316]:


import pandas as pd


# In[317]:


train = pd.read_csv("C:/Users/HP/Downloads/titanic/train.csv")
train.head()


# In[318]:


test = pd.read_csv("C:/Users/HP/Downloads/titanic/test.csv")
passengerId = test["PassengerId"]
test.head()


# In[319]:


train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis = 1)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis = 1)


# In[320]:


ftrs = ["Age", "Fare"]
for ftr in ftrs:
    train.fillna({ftr:train[ftr].mean()},inplace = True)
    test.fillna({ftr:test[ftr].mean()},inplace = True)
train.head()


# In[321]:


fltrs = ["Pclass","SibSp","Parch"]
for fltr in fltrs:
    train.fillna({fltr:train[fltr].median()},inplace = True)
    test.fillna({fltr:test[fltr].median()},inplace = True)
train.head(10)


# In[322]:


train.fillna({"Embarked":"N"},inplace = True)
test.fillna({"Embarked":"N"},inplace = True)
train.head()


# In[323]:


test.head()


# In[324]:


fare_mean_train = train["Fare"].mean()
fare_mean_test = test["Fare"].mean()
print(fare_mean_train)
print(fare_mean_test)


# In[325]:


age_mean_train = train["Age"].mean()
age_mean_test = test["Age"].mean()
print(age_mean_train)
print(age_mean_test)


# In[326]:


train.shape


# In[327]:


def_vals = [0 for _ in range(891)]
train.insert(8,"age_grtr_thn_mean",def_vals)
train.insert(9,"fare_grtr_thn_mean",def_vals)
train.head()


# In[328]:


test.shape


# In[329]:


def_vals = [0 for _ in range(418)]
test.insert(7,"age_grtr_thn_mean",def_vals)
test.insert(8,"fare_grtr_thn_mean",def_vals)
test.head()


# In[330]:


train.loc[train.Age <= age_mean_train, "age_grtr_thn_mean"] = 0
train.loc[train.Age > age_mean_train, "age_grtr_thn_mean"] = 1
train.loc[train.Fare <= fare_mean_train, "fare_grtr_thn_mean"] = 0
train.loc[train.Fare > fare_mean_train, "fare_grtr_thn_mean"] = 1
train.head()


# In[331]:


test.loc[test.Age <= age_mean_test, "age_grtr_thn_mean"] = 0
test.loc[test.Age > age_mean_test, "age_grtr_thn_mean"] = 1
test.loc[test.Fare <= fare_mean_test, "fare_grtr_thn_mean"] = 0
test.loc[test.Fare > fare_mean_test, "fare_grtr_thn_mean"] = 1
test.head(20)


# In[332]:


train.drop(["Age","Fare"], axis = 1)


# In[333]:


test.drop(["Age","Fare"], axis = 1)


# In[334]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
ftrs = ["Sex", "Embarked"]
for ftr in ftrs:
    train[ftr] = label_encoder.fit_transform(train[ftr])
    test[ftr] = label_encoder.transform(test[ftr])


# In[335]:


train.head()


# In[336]:


test.head()


# In[337]:


Y_train = train['Survived']
X_train = train.drop(["Survived"], axis = 1)


# In[338]:


X_train


# In[339]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, Y_train)
predict = knn.predict(test)
predict


# In[340]:


test_predict = pd.read_csv("C:/Users/HP/Downloads/titanic/gender_submission.csv")
test_predict = test_predict.drop(["PassengerId"], axis = 1)
test_predict


# In[341]:


print(f"Accuracy: {knn.score(test,test_predict)}")


# In[342]:


df = pd.DataFrame({"PassengerId":passengerId,"Survived":predict})
df.to_csv("C:/Users/HP/Downloads/titanic/submission.csv",index=False)

