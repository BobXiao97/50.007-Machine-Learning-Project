import pandas as pd

data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
observation=data['word']
state=data['state']


def emission_parameter(data):
    data1=data['state'].value_counts().reset_index(name='CountY')
    data1=data1.rename(columns={'index':'state'})
    data2=data.groupby(['word','state']).size().reset_index(name='CountY_X')
    data=pd.merge(data1,data2)
    data=data[['word','state','CountY','CountY_X']]
    data['emission']=data['CountY_X']/data['CountY']
    return data

states=list(set(state))
states.remove('Nil')
UNK_list=[]
for i in range(0,len(states)):
    UNK_list.append('#UNK#')
dic={'word':UNK_list,'state':states}
data_UNK=pd.DataFrame(dic)
def emission_parameter_UNK(data):
    data=pd.concat([data,data_UNK])
    data1=data['state'].value_counts().reset_index(name='CountY')
    data1=data1.rename(columns={'index':'state'})
    data2=data.groupby(['word','state']).size().reset_index(name='CountY_X')
    data=pd.merge(data1,data2)
    data=data[['word','state','CountY','CountY_X']]
    data['CountY+k']=data['CountY']-0.5
    data['emission']=data['CountY_X']/data['CountY+k']
    for i in range(0,len(data['word'])):
        if data['word'][i]=='#UNK#':
            data['emission'][i]=data['emission'][i]/2
    return data

def prediction(test,data):
    test=test['word']
    output=[]
    for i in range(0,len(test)):
        if test[i] not in data['word'].to_list():
            test[i]="#UNK#"
    for j in range(0,len(test)):
        data1=data.loc[data['word']==test[j]]
        emission=0
        state=''
        for index,row in data1.iterrows():
            if row['emission']>emission:
                emission=row['emission']
                state=row['state']
        output.append(state)
    return output

dev_in=pd.read_csv('dev.in',sep=' ',names=['word'])               
train_set=emission_parameter_UNK(data)
result=prediction(dev_in,train_set)

with open('dev.p2.out','w') as f:
    for i in range(0,len(result)):
        f.write(result[i]+'\n')
f.close
print('done')
    