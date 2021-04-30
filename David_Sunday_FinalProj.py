#!/usr/bin/env python
# coding: utf-8

# In[294]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics #neighbors, preprocessing, 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


from sklearn.preprocessing import LabelEncoder


# In[2]:


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


sns.set()
sns.pairplot(df, hue='class',height=3)
plt.show()


# In[6]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,:-1], 
                                                    df.iloc[:,-1], test_size=0.25)


# In[7]:


df.iloc[:,-1].unique()


# In[8]:


y_train[y_train.iloc[0:] == 'Iris-versicolor'] = 'r'
y_train[y_train.iloc[0:] == 'Iris-virginica'] = 'g'
y_train[y_train.iloc[0:] == 'Iris-setosa'] = 'b'


# In[9]:


plt.figure(figsize=(10,10))
plt.scatter(X_train.iloc[:,0:1], X_train.iloc[:,3:4], c=y_train[:], s=350, cmap='viridis')
plt.title('Training data')
plt.show()


# In[10]:


y_test[y_test.iloc[0:] == 'Iris-versicolor'] = 'r'
y_test[y_test.iloc[0:] == 'Iris-virginica'] = 'g'
y_test[y_test.iloc[0:] == 'Iris-setosa'] = 'b'

plt.figure(figsize=(10,10))
plt.scatter(X_test.iloc[:,0:1], X_test.iloc[:,3:4], c=y_test[:], s=350, cmap='viridis')
plt.title('Test data')
plt.show()


# In[11]:


y_train[y_train.iloc[0:] == 'r'] = 'Iris-versicolor'
y_train[y_train.iloc[0:] == 'g'] = 'Iris-virginica'
y_train[y_train.iloc[0:] == 'b'] = 'Iris-setosa'

y_test[y_test.iloc[0:] == 'r'] = 'Iris-versicolor'
y_test[y_test.iloc[0:] == 'g'] = 'Iris-virginica'
y_test[y_test.iloc[0:] == 'b'] = 'Iris-setosa'


# In[ ]:





# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB


# In[13]:


rf =RandomForestClassifier(n_estimators=100, random_state=101)


# In[14]:


knn = neighbors.KNeighborsClassifier()


# In[15]:


nb = GaussianNB()


# In[19]:


kf = KFold(n_splits=10)
kf


# In[202]:


def evaluate_CM(actual, pred):
    [[TPa, Eba, Eca],
    [Eab, TPb, Ecb],
    [Eac, Ebc, TPc]]= np.zeros((3,3))
    
    for i in range(len(actual)):
        if actual[i] == pred[i] and actual[i] == 0:
            TPa += 1
        if actual[i] == 1 and pred[i] == 0:
            Eba += 1
        if actual[i] == 2 and pred[i] == 0:
            Eca += 1
        if actual[i] == pred[i] and actual[i] == 1:
            TPb += 1
        if actual[i] == 0 and pred[i] == 1:
            Eab += 1
        if actual[i] == 2 and pred[i] == 1:
            Ecb += 1
            
        if actual[i] == pred[i] and actual[i] == 2:
            TPc += 1
        if actual[i] == 0 and pred[i] == 2:
            Eac += 1
        if actual[i] == 1 and pred[i] == 2:
            Ebc += 1 
    return np.array([[TPa, Eba, Eca],
                [Eab, TPb, Ecb],
                [Eac, Ebc, TPc]])  
      


# In[203]:


X = df.iloc[:,:-1]
Y = LabelEncoder().fit_transform(df.iloc[:,-1])
rf_cm = []
knn_cm = []
nb_cm = []


# In[204]:


def getCM(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_CM(y_test, y_pred)
    


# In[205]:


for train_index, test_index in kf.split(X, Y):
    rf_cm.append(getCM(rf, X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]))
    knn_cm.append(getCM(knn, X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]))
    nb_cm.append(getCM(nb, X.iloc[train_index], X.iloc[test_index], Y[train_index], Y[test_index]))


# In[206]:


knn_cm = np.divide(np.sum(knn_cm, axis=0), len(knn_cm))
rf_cm = np.divide(np.sum(rf_cm, axis=0), len(rf_cm))
nb_cm = np.divide(np.sum(nb_cm, axis=0), len(nb_cm))


# In[207]:


rf_cm


# In[208]:


knn_cm


# In[209]:


nb_cm


# In[221]:


def getAccuracy(cm):
    [[TPa, Eba, Eca],
    [Eab, TPb, Ecb],
    [Eac, Ebc, TPc]]=cm
    return (TPa+TPb+TPc)/sum(cm.flatten())


# In[222]:


def getPrecision(cm, index):
    return getTP(cm, index)/(getTP(cm, index) + getFP(cm, index))    


# In[240]:


def getTN(cm, index):
    tn = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if j != index and i != index:
                tn += cm[i][j]
    return tn


# In[224]:


def getTP(cm, index):
    return cm[index][index]


# In[225]:


def getFP(cm, index):
    fp = 0
    for i in range(len(cm)):
        if i != index:
            fp += cm[i][index]
    return fp


# In[226]:


def getFN(cm, index):
    fn = 0
    for i in range(len(cm)):
        if i != index:
            fn += cm[index][i]
    return fn


# In[227]:


def getTPR(cm, index): #sensitivity/RECALL
    return getTP(cm, index)/(getTP(cm, index) + getFN(cm, index))


# In[228]:


def getTNR(cm, index): #Specificity
    return getTN(cm, index)/(getTN(cm, index) + getFP(cm, index))


# In[269]:


def getF1Score(cm, i):
    return 2*(getPrecision(cm, i) * getTPR(cm, i))/(getPrecision(cm, i) + getTPR(cm, i))


# In[258]:


classifiers = [rf_cm, knn_cm, nb_cm]
labels = ["Random Forest", "KNN", "Naive Bayes"]


# In[ ]:





# In[259]:


classes = LabelEncoder().fit(df.iloc[:,-1]).classes_


# In[330]:


df2 = []
Acc = []
for i in range(len(classifiers)):
    TP = []
    FP = []
    TN = []
    FN = []
    TP_FN = []
    FP_TN = []
    Prec = []
    Spec = []
    Score = []
    Sens = []
    for j in range(len(classes)):
        TP.append(getTP(classifiers[i], j))
        FP.append(getFP(classifiers[i], j))
        TN.append(getTN(classifiers[i], j))
        FN.append(getFN(classifiers[i], j))
        TP_FN.append(TP[j] + FN[j])
        FP_TN.append(FP[j] + TN[j])
        Prec.append(getPrecision(classifiers[i], j))
        Spec.append(getTNR(classifiers[i], j))
        Score.append(getF1Score(classifiers[i], j))
        Sens.append(getTPR(classifiers[i], j))
    Acc.append(getAccuracy(classifiers[i])) 
    df2.append(pd.DataFrame({"TP":TP, "FP":FP, "TN":TN, "FN":FN, "TP+FN":TP_FN, "FP+TN":FP_TN, "Pecision":Prec, 
                  "Specificity":Spec, "Score":Score, "Sensitivity":Sens}, index=classes))


# In[332]:


for i in range(len(df2)):
    print(labels[i], "Accuracy: ", Acc[i])
    print(df2[i])
    print('\n')


# In[ ]:




