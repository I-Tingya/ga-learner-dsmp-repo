# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
for i in range(len(cols)):
    df[cols[i]]= df[cols[i]].str.replace('$','').str.replace(',','')

X= df.drop('CLAIM_FLAG',axis=1)
y = df.CLAIM_FLAG

count = y.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=6)

# Code ends here


# --------------
# Code starts here
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM']

X_train[cols] = X_train[cols].astype('float64')
X_test[cols]  = X_test[cols].astype('float64')

print(X_train.isnull().sum())
print(X_test.isnull().sum())
# Code ends here


# --------------
# Code starts here

X_train = X_train.dropna(subset=['YOJ','OCCUPATION'])
X_test = X_test.dropna(subset=['YOJ','OCCUPATION'])
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
cols = ['AGE','CAR_AGE','INCOME','HOME_VAL']
for i in range(len(cols)):
    X_train[cols[i]] = X_train[cols[i]].fillna(X_train[cols[i]].mean()) 
    X_test[cols[i]] = X_test[cols[i]].fillna(X_test[cols[i]].mean()) 
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for i in range(len(columns)):
    le.fit(X_train[columns[i]])
    X_train[columns[i]]=le.transform(X_train[columns[i]].astype(str))
    X_test[columns[i]]=le.transform(X_test[columns[i]].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred) 
precision = precision_score(y_test,y_pred)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

smote = SMOTE(random_state=6)
X_train,y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# Code ends here


# --------------
# Code Starts here

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred) 
# Code ends here


