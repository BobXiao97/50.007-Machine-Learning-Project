import pandas as pd

data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
observation=data['word']
state=data['state']
states=list(set(state))

trans_p=pd.read_csv('transition parameter.csv',index_col=0)
emi_p=pd.read_csv('emission parameter.csv',index_col=0)
#print(emi_p[0])
#print(emi_p[1])
for y in states:

    print(trans_p[y])