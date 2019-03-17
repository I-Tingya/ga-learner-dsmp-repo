# --------------
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv(path,sep='\t')
print(df.shape)
# Converting date attribute from string to datetime.date datatype 
print(df['date'].dtype)
df['date'] = df['date'].astype(datetime.date)

# calculate the total length of word
df['length'] = df['verified_reviews'].str.len()



# --------------
## Rating vs feedback

# set figure size
plt.figure(figsize=(10,6))

# generate countplot
sns.countplot(x='rating',hue='feedback',data=df)

# display plot
plt.show()

## Product rating vs feedback

# set figure size
plt.figure(figsize=(10,6))

# generate barplot
sns.barplot(x='rating',hue='feedback',y='variation',data=df)

# display plot
plt.show()



# --------------
# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# declare empty list 'corpus'
corpus=[]

# for loop to fill in corpus
for i in range(3150):
    # retain alphabets
    review = re.findall(r'\w+',df.loc[i,'verified_reviews'])
    # convert to lower case
    review = [txt.lower() for txt in review]
    # tokenize
    #review = review.split()
    # initialize stemmer object
    ps = PorterStemmer()
    # perform stemming
    stop_words=set(stopwords.words('english'))
    review = [ps.stem(i) for i in review if i not in stop_words]
    # join elements of list
    review = ' '.join(review)
    # add to 'corpus'
    corpus.append(review)
    
# display 'corpus'

corpus



# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
cv = CountVectorizer(max_features=1500)

# Independent variable
X = cv.fit_transform(corpus)

# dependent variable
y = df.feedback

# Counts
count = y.value_counts()

# Split the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)



# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Instantiate calssifier
rf = RandomForestClassifier(random_state=2)

# fit model on training data
rf.fit(X_train,y_train)

# predict on test data
y_pred = rf.predict(X_test) 

# calculate the accuracy score
score = accuracy_score(y_test,y_pred)

# calculate the precision
precision = precision_score(y_test,y_pred)

# display 'score' and 'precision'



# --------------
# import packages
from imblearn.over_sampling import SMOTE

# Instantiate smote
smote = SMOTE()

# fit_sample onm training data
X_train,y_train = smote.fit_sample(X_train,y_train)

# fit modelk on training data
rf.fit(X_train,y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test,y_pred)

# calculate the precision
precision = precision_score(y_test,y_pred)

# display precision and score
print(score,precision)


