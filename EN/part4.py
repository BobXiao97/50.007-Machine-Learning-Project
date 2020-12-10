# for quick access, to get the location of each tags in the dictionary
dic = {'START': 0, 'B-ADJP': 1, 'B-ADVP': 2, 'B-CONJP': 3, 'B-INTJ': 4, 'B-LST': 5, 'B-NP': 6,
       'B-PRT':7,'B-SBAR':8,'B-UCP':9,'B-VP':10,'I-ADJP':11,'I-ADVP':12,'I-CONJP':13,'I-INTJ':14,'I-LST':15,
       'I-NP':16,'I-PP':17,'I-PRT':18,'I-SBAR':19,'I-UCP':20,'I-VP':21,'O': 22, 'STOP': 23}
l = ['START', 'B-ADJP', 'B-ADVP', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-NP',
       'B-PRT','B-SBAR','B-UCP','B-VP','I-ADJP','I-ADVP','I-CONJP','I-INTJ','I-LST',
       'I-NP','I-PP','I-PRT','I-SBAR','I-UCP','I-VP','O', 'STOP']

def train():

    # store emission parameters
    e_count = ({}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {})  ## 1st dict empty (start no emission)
    e_param = ({}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {})  ## 1st dict empty (start no emission)

    # store transition parameters
    # initialize as a 25*24 matrix of zeros
    w, h = 25, 24
    t_param = [[0] * w for i in range(h)]

    count = [0] * 24

    ## read and parse file
    train_file = open('train', 'r')
    u = 'START'
    obs_space = set()

    for obs in train_file:
        try:
            obs, v = obs.split()
            obs = obs.strip()
            v = v.strip()
            position = dic[v]  ## position: 1~23
            # update e_count
            if (obs in e_count[position]):
                e_count[position][obs] += 1
            else:
                e_count[position][obs] = 1

            # update t_param
            pre_position = dic[u]
            t_param[pre_position][position] += 1
            u = v

            # add into train_obs_set
            if obs not in obs_space:
                obs_space.add(obs)

        except:
            # meaning the end of a sentence: x->STOP
            pre_position = dic[u]
            t_param[pre_position][24] += 1
            u = 'START'

    # get count(yi)+1
    for i in range(0, 24):
        temp_sum = 0
        for j in range(0, 25):
            temp_sum = temp_sum + t_param[i][j]
        count[i] = temp_sum + 1

    ## convert transision param to probablity
    for i in range(0, 24):
        for j in range(0, 24):
            t_param[i][j] = 1.0 * t_param[i][j] / count[i]

    # building emission params table: a list of 8 dicts, each dict has all obs as keys,
    # value is 0 if obs never appears for this state

    for i in range(1,23):
        for obs in obs_space:
            if obs not in e_count[i]:
                e_param[i][obs] = 0.5 / max(count)
            else:
                e_param[i][obs] = 1.0 * e_count[i][obs] / count[i]

    print(t_param)
    return obs_space, e_param, t_param, count


def forwardalgo(prev_layer, x, k):
       # inputs: prev_layer: list of list of top k best, x: current word, k: top k best
       # output: list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states, len=7

    layer = []
    for i in range(1, 23):  #
        temp_score = []
        states = []
        n = len(prev_layer[0])
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, 23):  # j:1-23
            for sub in range(0, n): # n scores for each prev_node
                # score = prev_layer*a*b
                j_score = prev_layer[j-1][sub][0] * (t_param[j][i]) * b
                temp_score.append([j_score, j-1, sub])  # 23*n scores with their parents
        temp_score.sort(key=lambda tup:tup[0],reverse=True) # sort by j_score
        for sub in range(0, k):   # get top k best
            states.append(temp_score[sub])
        layer.append(states)
    return layer


def viterbi(X, k):
       # input:  X: words list, k: top k best
       # output: Y: tag list

    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # calculate layer (start ->) 1
    x = X[0]
    for j in range(1, 23):
        state = []
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        state.append([prob, 0, 0])  # [prob, START, 1st best]
        prev_layer.append(state)
    layers = [[(1, -1, 0)], prev_layer]


    # calculate layer (2,...,n)
    for i in range(1, n):  # prev_layer: 1 -> n-1
        layer = forwardalgo(layers[i], X[i], k)  # a list of top k best scores
        layers.append(layer)


    # calculate layer n+1 (STOP), and get top k best
    layer = []
    temp_score = []
    states = []
    failed = False
    for j in range(1, 23):  # j:1-23
        for sub in range(0, len(layers[n][0])):  # kth score for each prev_node
            # score = prev_layer*a
            t_score = layers[n][j - 1][sub][0] * (t_param[j][8])
            temp_score.append([t_score, j - 1, sub])  # 23*k scores with their parents

    temp_score.sort(key=lambda tup: tup[0], reverse=True)  # sort by j_score
    for sub in range(0, k):  # get top k best
        states.append(temp_score[sub])
    layer.append(states)
    layers.append(layer)

    # backtracking
    parent_index = 0    # only 1 state in STOP
    parent_sub = k-1   # kth best score in STOP layer
    for i in range(n+1, 1, -1):  # index range from N to 2
        a = layers[i][parent_index][parent_sub][1]
        b = layers[i][parent_index][parent_sub][2]
        Y.insert(0, l[a + 1])  # 1-23
        parent_index = a
        parent_sub = b
    return Y


def viterbiTopK(obs_space, e_param, t_param, count, k):
    dev_file = open('dev.in', 'r')
    out_file = open('dev.p4.out', 'w')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbi(X, k)
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)


obs_space, e_param, t_param, count = train()
k = 3 ## top 3 best
viterbiTopK(obs_space, e_param, t_param, count, k)
