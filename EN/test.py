import pandas as pd
data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
observation=data['word']
state=data['state']
states=list(set(state))

trans_p=pd.read_csv('transition parameter.csv',index_col=0)
emi_p=pd.read_csv('emission parameter.csv',index_col=0)

v=[]
vn=emi_p['Nil']
v.append(vn)
w=[]
tp=emi_p['HBO']
gv=v[0]*trans_p['B-NP']*tp['B-NP']
x=gv.max()
y=gv.to_list()
index=y.index(max(y))
print(index)