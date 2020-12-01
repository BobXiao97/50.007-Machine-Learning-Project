import pandas as pd
trans_p=pd.read_csv('transition parameter.csv',index_col=0)
emi_p=pd.read_csv('emission parameter.csv')
print(trans_p)