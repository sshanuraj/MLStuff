import numpy as np
import random as rd

def select_random(k):
    option = int(np.random.uniform(0, k))
    return option

def select_greedy(qvalues):
    max_index = np.argmax(qvalues)
    return max_index

def get_reward(index):
    if index%3 == 0:
        return 5
    return -1

def run_bandit(n, k, eps):
    qvalues = np.zeros(k,)
    for i in range(n):
        prob = np.random.uniform(0, 1)
        index = 0
        if eps >= prob:
            index = select_random(k)
        else:
            index = select_greedy(qvalues)

        reward = get_reward(index)
        qvalues[index] = qvalues[index] + ((1/(i+1))*(reward - qvalues[index]))
    return qvalues

print(run_bandit(10000, 10, 0.1))


