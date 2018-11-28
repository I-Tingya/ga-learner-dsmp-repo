# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best = data[data['Total']>total_high]
super_best_names = super_best['Name'].tolist()
print(super_best_names)


# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
#Code starts here 
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
plt.bar(gender_count.index, gender_count)



# --------------
#Code starts here
alignment = data['Alignment'].value_counts()

plt.pie(alignment)

plt.title('Character Alignment')


# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']].copy()
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])
#print(sc_covariance)
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_combat*sc_strength)

ic_df = data[['Intelligence', 'Combat']].copy()
ic_covariance = ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = ic_covariance/(ic_combat*ic_intelligence)


# --------------
#Code starts here
fig, (ax_1,ax_2,ax_3) = plt.subplots(3,1)

ax_1.plot(data['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.plot(data['Speed'])
ax_2.set_title('Speed')
ax_3.plot(data['Power'])
ax_3.set_title('Power')


