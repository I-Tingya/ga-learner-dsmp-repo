# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
dataset.head()

# Check if there's any column which is not useful and remove it like the column id
dataset.drop('Id',axis=1,inplace=True)

# check the statistical description

print(dataset.info())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns
#print(cols)
#number of attributes (exclude target)
size = len(cols)-1
#print(size)
#x-axis has target attribute to distinguish between classes
x = dataset.Cover_Type
#x.replace({1:'Spruce/Fir',2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Crummholz'},inplace = True)

print(x.dtype)
#y-axis shows values of an attribute
y = dataset.drop('Cover_Type',axis=1)

#Plot violin for all attributes
'''
p=0
figure,axes = plt.subplots(nrows=18,ncols=3,figsize=(10,100))
for i in range(18):
    for j in range(3):
        axes[i][j].violinplot(dataset.iloc[:,p])
        p = p + 1
'''
figure,axes = plt.subplots(54,figsize=(10,100))
for i in range(size):
    sns.violinplot(data=y,x=x,y=y.iloc[:,i],ax=axes[i])

plt.show()


# --------------
import numpy
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'

subset_train = dataset.iloc[:, :num_features]
cols = subset_train.columns

#Calculate the pearson co-efficient for all possible combinations

data_corr = subset_train.corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(data_corr,vmax=0.8,square=True);

# Set the threshold and search for pairs which are having correlation level above threshold
corr_var_list = []

for i in range(0, num_features):
    for j in range(i+1, num_features):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_var_list.append([data_corr.iloc[i,j], i, j])

# Sort the list showing higher ones first 
s_corr_list = sorted(corr_var_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
y = dataset.Cover_Type
X = dataset.drop('Cover_Type',axis=1)


# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)
scaler = StandardScaler()

#Standardized
#Apply transform only for non-categorical data
X_train_temp = scaler.fit_transform(X_train.iloc[:,0:11])
X_test_temp = scaler.transform(X_test.iloc[:,0:11])

#Concatenate non-categorical data and categorical
X_train1 = pd.DataFrame(X_train_temp)
X_test1 = pd.DataFrame(X_test_temp)
scaled_features_train_df = pd.concat([X_train1.iloc[:,0:11],X_train.iloc[:,11:]], axis=1).reindex_like(X_train)
scaled_featured_test_df = pd.concat([X_test1.iloc[:,0:11],X_test.iloc[:,11:]],axis=1).reindex_like(X_test)

#X_train1.head()


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(score_func=f_classif,percentile=20)

predictors = skb.fit(X_train1,Y_train)

scores = predictors.scores_
pval = predictors.pvalues_
#predictors = skb.fit(X_train1,Y_train)

top_k_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)#[:predictors.shape[1]]
top_k_predictors = [scaled_features_train_df.columns[i] for i in top_k_index]
print(top_k_index)
print(top_k_predictors)





# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
clf = OneVsRestClassifier(LogisticRegression())
clf1 = OneVsRestClassifier(LogisticRegression())

model_fit_all_features = clf1.fit(X_train, Y_train)

predictions_all_features = model_fit_all_features.predict(X_test)

score_all_features = accuracy_score(Y_test, predictions_all_features)

print(score_all_features)

model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)

predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])

score_top_features = accuracy_score(Y_test, predictions_top_features)

print(score_top_features)


