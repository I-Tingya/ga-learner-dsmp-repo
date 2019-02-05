# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

df = pd.read_csv(path)
# Code starts here
X = df.drop(['customerID','Churn'],axis=1)
y = df.Churn
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)



# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here
X_train.TotalCharges = X_train.TotalCharges.replace({' ':np.NaN})
X_test.TotalCharges = X_test.TotalCharges.replace({' ':np.NaN})

X_train.TotalCharges = X_train.TotalCharges.astype('float')
X_test.TotalCharges = X_test.TotalCharges.astype('float')

X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(),inplace=True)
X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(),inplace=True)

print(X_train.isnull().sum())

cat_var = X_train.select_dtypes(include='object').columns.values.tolist()
le = LabelEncoder()
for i in range(0,len(cat_var)):
    le.fit(X_train[cat_var[i]])
    X_train[cat_var[i]] = le.transform(X_train[cat_var[i]])
    X_test[cat_var[i]] = le.transform(X_test[cat_var[i]])

y_train = y_train.replace({'No':0,'Yes':1})
y_test = y_test.replace({'No':0,'Yes':1})



# --------------
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
# Code starts here
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred = ada_model.predict(X_test)

ada_score = accuracy_score(y_test,y_pred)
print(ada_score)

ada_cm = confusion_matrix(y_test,y_pred)
print(ada_cm)

ada_cr = classification_report(y_test,y_pred)
print(ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)

y_pred = xgb_model.predict(X_test)

xgb_score = accuracy_score(y_test,y_pred)
xgb_cm = confusion_matrix(y_test,y_pred)
xgb_cr = classification_report(y_test,y_pred)

xgb_clf = XGBClassifier(random_state=0)
clf_model = GridSearchCV(estimator=xgb_clf,param_grid=parameters)

clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test,y_pred)
clf_cm = confusion_matrix(y_test,y_pred)
clf_cr = classification_report(y_test,y_pred)
print(clf_model.best_estimator_)
print((xgb_score,clf_score))
print(xgb_cm)
print(clf_cm)
print(xgb_cr)
print(clf_cr)



