# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Code starts here
data = pd.read_csv(path)
#data.head()
data=data[data['Rating']<=5]
plt.hist(data['Rating'])

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()

percent_null = total_null/total_null.count()

missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total','Percent'])

#print(missing_data)

data = data.dropna(axis=0)

total_null_1 = data.isnull().sum()

percent_null_1 = total_null_1/total_null_1.count()

missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total','Percent'])
# code ends here


# --------------

#Code starts here

sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data['Installs'] = data['Installs'].str.replace(',','').str.replace('+','')

data['Installs'] = data['Installs'].astype('int')
le = LabelEncoder()
le.fit(data['Installs'])
data['Installs']=le.transform(data['Installs'])

sns.regplot(x='Installs', y='Rating', data=data)
plt.title('Rating Vs Installs [RegPlot]')
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())

data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype('float')

sns.regplot(x='Price', y='Rating', data=data) 
plt.title('Rating Vs Price [RegPlot]')
#Code ends here


# --------------

#Code starts here

print(data['Genres'].unique())

data['Genres'] = data['Genres'].str.split(pat=';', n=1, expand=True)[0]

gr_mean = data.groupby('Genres',as_index=False)['Genres','Rating'].mean()

print(gr_mean.describe())

gr_mean = gr_mean.sort_values(by='Rating')

print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
print(data['Last Updated'])

data['Last Updated']= pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
print(max_date)

data['Last Updated Days'] = max_date - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days

sns.regplot(x='Last Updated Days', y='Rating',data=data)
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here


