import pandas as pd

data=pd.read_csv('train',sep=' ',names=['word','state'],skip_blank_lines=False)
data=data.fillna('Nil')
observation=data['word']
state=data['state']
possible_state=list(set(state))


def emission_parameter(x,y): # x is words, y is tags
    count_y_x=0
    count_y=0
    for i in range(0,len(state)):
        if state[i]==y:
            count_y+=1
            if observation[i]==x:
                count_y_x+=1
    return count_y_x/count_y


def emission_parameter_UNK(test,y): # y is tags, test is the test data that might not be in the training set
    count_y_test=0
    count_y=0
    if test in observation:
        for i in range(0,len(state)):
            if state[i]==y:
                count_y+=1
                if observation[i]==test:
                    count_y_test+=1
        result=count_y_test/(count_y+0.5)
    else:
        for i in range(0,len(state)):
            if state[i]==y:
                count_y+=1
        result=0.5/(count_y+0.5)
    return result


def emission_prediction(test):
    possibility=[]
    for i in range(0,len(possible_state)):
        p=emission_parameter_UNK(test,possible_state[i])
        possibility.append(p)      
    max_index=possibility.index(max(possibility))
    return possible_state[max_index]


dev_in=pd.read_csv('dev.in',sep=' ',names=['input'])               
test_x=dev_in['input']
test_x_prediction=[]
for i in range(0,len(test_x)):
    prediction=emission_prediction(test_x[i])
    test_x_prediction.append(prediction)

with open('dev.p2.out','w') as f:
    for i in range(0,len(test_x_prediction)):
        f.write(test_x_prediction[i]+'\n')
f.close


    
    
           