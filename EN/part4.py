import pandas as pd
import numpy as np
import copy

possible_states = ["O","B-ADJP", "B-ADVP", "B-CONJP", "B-INTJ", "B-LST", "B-NP","B-PP","B-PRT","B-SBAR",
                   "B-UCP","B-VP","I-ADJP", "I-ADVP", "I-CONJP", "I-INTJ", "I-LST", "I-NP","I-PP","I-PRT",
                   "I-SBAR","I-UCP","I-VP","Nil"]

def viterbi(obs, states, trans_p, emit_p):
    v = []
    fst = obs[0]
    vn = emit_p[fst]
    v.append(vn)

    for t in range(1, len(obs)):
        tp = emit_p[obs[t]]
        cc = []
        state_pos = ""
        val = 0
        for y in states:
            print(y)
            gv = v[t - 1] * trans_p[y] * tp.loc[y]
            cc.append(gv.max())
        cc1 = pd.Series(cc, index=states)
        v.append(cc1)

    result = []
    for vector in v:
        p = vector
        p1 = p.sort_values(ascending=False)
        p2 = p1[:1]
        result.append(dict(p2))
    return result


data = pd.read_csv('train', sep=' ', names=['word', 'state'], skip_blank_lines=False)
data = data.fillna('Nil')
observation = data['word']
state = data['state']
states = list(set(state))

dev_in = pd.read_csv('dev.in', sep=' ', names=['word'], skip_blank_lines=False)
dic = {'word': ['Nil']}
first_line = pd.DataFrame(dic)
dev_in = pd.concat([first_line, dev_in], ignore_index=True)
dev_in_list = np.split(dev_in, dev_in[dev_in.isnull().all(1)].index)
for i in range(1, len(dev_in_list)):
    dev_in_list[i] = dev_in_list[i].fillna('Nil')
for j in range(0, len(dev_in_list) - 1):
    dev_in_list[j] = pd.concat([dev_in_list[j], first_line], ignore_index=True)

for i in range(0, len(dev_in_list)):
    dev_in_list[i] = dev_in_list[i]['word'].to_list()
    for j in range(0, len(dev_in_list[i])):
        if dev_in_list[i][j] not in observation.to_list():
            dev_in_list[i][j] = '#UNK#'
trans_p = pd.read_csv('transition parameter.csv', index_col=0)
emi_p = pd.read_csv('emission parameter.csv', index_col=0)

key = []
for i in range(0, len(dev_in_list)):
    result = viterbi(dev_in_list[i], states, trans_p, emi_p)
    for j in range(0, len(result)):
        k = list(result[j].keys())
        key.append(k[0])

with open('dev.p4.out', 'w') as f:
    for i in range(0, len(key)):
        f.write(key[i] + '\n')
f.close

def emis_prob(state, word, training_data, emis_dict):
    if (state, word) in emis_dict.keys():
        return emis_dict[(state, word)]
    else:
        count_emission = 0 # count the emission from state to word
        count_state = 1
        count_word = 0

        for tweet in training_data:
            for j in range(len(tweet)):
                if tweet[j].split(" ")[0] == word:
                    count_word += 1
                if tweet[j].split(" ")[1] == state:
                    count_state += 1
                    if tweet[j].split(" ")[0] == word:
                        count_emission += 1

        if count_word == 0:
            result =  float(1/count_state)
        else:
            result = float(count_emission/count_state)

        emis_dict[(state, word)] = result
        return result


def trans_prob(state1, state2, training_data, trans_dict):
    if (state1, state2) in trans_dict.keys():
        return trans_dict[(state1, state2)]
    else:
        count_transition = 0 # count the transition from state1 to state 2
        count_state1 = 0

        if state1 == 'start':
            count_state1 = len(training_data)
            for tweet in training_data:
                if len(tweet[0].split(" ")) > 1 and tweet[0].split(" ")[1] == state2:
                    count_transition += 1

        elif state2 == 'stop':
            for tweet in training_data:
                for j in range(len(tweet)):
                    if len(tweet[j].split(" ")) > 1  and tweet[j].split(" ")[1] == state1:
                        count_state1 += 1
                        if j == len(tweet) - 1:
                            count_transition += 1

        elif state1 == 'stop' or state2 == 'start':
            trans_dict[(state1, state2)] = 0
            return 0

        else:
            for tweet in training_data:
                for j in range(len(tweet)-1):
                    if len(tweet[j].split(" ")) > 1 and tweet[j].split(" ")[1] == state1:
                        count_state1 += 1
                        if tweet[j+1].split(" ")[1] == state2:
                            count_transition += 1

        result = float(count_transition/count_state1)
        trans_dict[(state1, state2)] = result
        return result


def viterbi_topK_kth_label(dev_datapath, training_datapath, k, os):
    trans_dict = {}
    emis_dict = {}
    training_data = Data_processor(training_datapath).data

    if os == "W":
        outpath = dev_datapath.rsplit("\\",maxsplit=1)[0] + "\\dev.p4.out"
    else:
        outpath = dev_datapath.rsplit("/",maxsplit=1)[0] + "/dev.p4.out"

    outfile = open(outpath, 'w', encoding='utf8')
    dev_data = Data_processor(dev_datapath).data
    total_tweets = len(dev_data)

    for tweet in range(total_tweets):
        score_dict = {}
        top_k_list = viterbi_topK_end(dev_data[tweet], k, emis_dict, trans_dict, training_data, score_dict)
        top_k_list = sorted(top_k_list, key=lambda x:x[1])
        kth_seq = top_k_list[0]
        tags = kth_seq[0].split(" ")

        for word in range(len(tags)):
            output = dev_data[tweet][word] + " " + tags[word] + "\n"
            outfile.write(output)

        outfile.write("\n")
        print(str(tweet+1) + "/" + str(total_tweets) + " done")

    print("Labelling completed!")
    outfile.close()


def viterbi_topK_start(sequence, state, emis_dict, trans_dict, training_data, score_dict):
    if (len(sequence), state) in score_dict.keys():
        return score_dict[(len(sequence), state)]
    else:
        score = trans_prob("start", state, training_data, trans_dict) * emis_prob(state, sequence[-1], training_data, emis_dict)
        score_dict[(len(sequence), state)] = [(state, score)]
        return [(state, score)]


def viterbi_topK_end(sequence, k, emis_dict, trans_dict, training_data, score_dict):
    top_k_list = []

    for state in possible_states:
        if len(sequence) == 1:
            previous_list = viterbi_topK_start(sequence, state, emis_dict, trans_dict, training_data, score_dict)
        else:
            previous_list = viterbi_topK_recursive(sequence, k, state, emis_dict, trans_dict, training_data, score_dict)

        for j in previous_list:
            score = j[1] * trans_prob(state, "stop", training_data, trans_dict)
            if score != 0:
                if len(top_k_list) < k:
                    top_k_list.append((j[0], score))
                else:
                    index = 0
                    for l in range(1, len(top_k_list)):
                        if top_k_list[l][1] < top_k_list[index][1]:
                            index = l
                    if score > top_k_list[index][1]:
                        top_k_list[index] = (j[0], score)

    if len(top_k_list) == 0:
        previous_O_list = viterbi_topK_recursive(sequence, k, "O", emis_dict, trans_dict, training_data, score_dict)
        top_k_list.append((previous_O_list[0][0],0))
    return top_k_list


def viterbi_topK_recursive(sequence, k, state, emis_dict, trans_dict, training_data, score_dict):
    if (len(sequence), state) in score_dict.keys():
        return score_dict[(len(sequence), state)]
    else:
        k_list = []
        for prev_state in possible_states:
            if len(sequence) == 2:
                previous_list = viterbi_topK_start(sequence[:-1], prev_state, emis_dict, trans_dict, training_data, score_dict)
            else:
                previous_list = viterbi_topK_recursive(sequence[:-1], k, prev_state, emis_dict, trans_dict, training_data, score_dict)

            for z in previous_list:
                score = z[1] * trans_prob(prev_state, state, training_data, trans_dict) * emis_prob(state, sequence[-1], training_data, emis_dict)
                y_seq = z[0] + " " + state
                if score != 0:
                    if len(k_list) < k:
                        k_list.append((y_seq, score))
                    else:
                        k_index = 0
                        for l in range(1, len(k_list)):
                            if k_list[l][1] < k_list[k_index][1]:
                                k_index = l
                        if score > k_list[k_index][1]:
                            k_list[k_index] = (y_seq, score)

        if len(k_list) == 0:
            previous_O_list = viterbi_topK_recursive(sequence[:-1], k, "O", emis_dict, trans_dict, training_data, score_dict)
            y_seq = previous_O_list[0][0] + " " + state
            k_list.append((y_seq, 0))

        score_dict[(len(sequence), state)] = k_list
        return k_list


if len(sys.argv) < 5:
    print("Not enough arguments pls input in order: (k-value, input data file path, training data file path, 'W'(for Windows) or 'L'(for Linux/Mac)")
    sys.exit()

viterbi_topK_kth_label(3,"dev.in",int(sys.argv[1]),sys.argv[4])