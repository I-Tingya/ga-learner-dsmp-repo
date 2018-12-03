# --------------
# code starts here
inst_median = df['installment'].median
inst_mean = df['installment'].mean 
df['installment'].hist()
#print(df.columns)
df['log.annual.inc'].hist()


# code ends here


# --------------
# code starts here

#plt.bar(df['purpose'].unique().index, df['purpose'].unique().value)
#plt.show()
df['purpose'].value_counts().plot.bar()
df1 = df[df['paid.back.loan']=='No']
df1['purpose'].value_counts().plot.bar()
# code ends here


# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
p_a = sum(df['fico']>700)/df.shape[0]
p_b = sum(df['purpose']=='debt_consolidation')/df.shape[0]
#print(df['purpose'].unique())
df1 = df[df['purpose']=='debt_consolidation']
p_a_b = sum(df1['fico']>700)/df.shape[0]
result = p_a_b == p_a
print(result)
# code ends here


# --------------
# code starts here
#print(df.columns)

prob_lp = sum(df['paid.back.loan']=='Yes')/df.shape[0]
prob_cs = sum(df['credit.policy']=='Yes')/df.shape[0]

new_df = df[df['paid.back.loan']=='Yes']
prob_pd_cs = (sum(new_df['credit.policy']=='Yes')/df.shape[0])/prob_lp

bayes = (sum(new_df['credit.policy']=='Yes')/df.shape[0])/prob_cs

print(bayes)
# code ends here


