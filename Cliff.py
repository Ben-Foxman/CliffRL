import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

#hyperparameters
nrows = 6
ncols = 14
nact = 4
nepisodes = 100000
epsilon = .1
alpha = .1
gamma = .5

reward_normal = -1
reward_cliff = -100
reward_destination = 50

#####

Q = np.zeros((nrows, ncols, nact), dtype = np.float64)
def go_to_start():
    x = 0
    y = nrows
    return x, y

x, y = go_to_start()
@jit(nopython=True)
def explore():
    # 0 = up
    # 1 = right
    # 2 = down
    # 3 = left
    a = np.random.randint(nact)
    return a


@jit(nopython=True)
def move(x, y, a):
    state = 0
    if x == 0 and y == nrows and a == 0:
        #start location
        x1 = x
        y1 = y - 1
        return x1, y1, state
    elif x == ncols - 1 and y == nrows-1 and a == 2:
        #reached destination
        x1 = x
        y1 = y + 1
        state = 1
        return x1, y1, state
    else:
        if a == 0:
            x1, y1 = x, y - 1
        elif a == 1:
            x1, y1 =  x+1, y
        elif a == 2:
            x1, y1 =  x, y+1
        elif a == 3:
            x1, y1 = x-1, y

    if x1 < 0:
        x1 = 0
    elif x1 > ncols - 1:
        x1 = ncols - 1
    if y1 < 0:
        y1 = 0
    elif y1 > nrows - 1:
        state = 2
    return x1, y1, state

#max Q val corresponding to move at a location

@jit(nopython=True)
def exploit(x, y, Q):
    if x == 0 and y == nrows:
        a = 0
        return a
    if x == ncols - 1 and y == nrows - 1:
        a = 2
        return a

    if x == ncols - 1 and y == nrows:
        print("Cannot exploit at when at destination.")
        return None

    if x < 0 or x > ncols - 1 or y < 0 or y > nrows - 1:
        print("Error, invalid location.", x, y)
        return None

    a = np.argmax(Q[y, x, :])
    return a

#update Q vals
@jit(nopython=True)
def bellman(x, y, a, reward, Qs1a1, Q):
    if x == 0 and y == nrows:
        return Q
    if x == ncols - 1 and y == nrows:
        return Q
    Q[y, x, a] += alpha * (reward + gamma*Qs1a1 - Q[y,x,a])
    return Q



def explore_exploit(x, y, Q):
    if x == 0 and y == nrows:
        a = 0
        return a
    r = np.random.uniform()
    if r < epsilon:
        #explore
        a = explore()
    else:
        a = exploit(x, y, Q)
    return a


#####################################################
###  Q-learning addition
def max_Q(x, y, Q):
    a = np.argmax(Q[y, x, :])
    return Q[y, x, a]



######################################################
######################################################


for i in range(nepisodes):
    if i % 100 == 99:
        print("Episode #:", i + 1)
    x, y = go_to_start()
    # SARSA: Explore/Exploit inside while loop.
    # a = explore_exploit(x, y, Q)
    while True :
        #Q-Learning: Explore/Exploit inside while loop.
        a = explore_exploit(x, y, Q)
        x1, y1, state = move(x, y, a)
        #Bellman Update
        if state == 1:
            reward = reward_destination
            Qs1a1 = 0.0
            Q = bellman(x, y, a, reward, Qs1a1, Q)
            break
        elif state == 2:
            reward = reward_cliff
            Qs1a1 = 0.0
            Q = bellman(x, y, a, reward, Qs1a1, Q)
            break
        elif state == 0:
            reward = reward_normal
            a1 = explore_exploit(x1, y1, Q)
            if x1 == 0 and y1 == nrows:
                Qs1a1 = 0.0
            else:
                #Q-Learning Variant
                Qs1a1 = max_Q(x1, y1, Q)
                #Sarsa Variant
                #Qs1a1 = Q[y1, x1, a1]
            Q = bellman(x, y, a, reward, Qs1a1, Q)
            x = x1
            y = y1
            a = a1

#Q-values converged to optimal
#plot

print(Q[:, :, 0])
print()
print(Q[:, :, 1])
print()
print(Q[:, :, 2])
print()
print(Q[:, :, 3])


for i in range(nact):
    plt.subplot(nact, 1, i + 1)
    plt.imshow(Q[:, :, i])
    plt.axis("Off")
    plt.colorbar(cmap=)
    if i == 0:
        plt.title("Q - North")
    elif i == 1:
        plt.title("Q - East")
    elif i == 2:
        plt.title("Q - South")
    elif i == 3:
        plt.title("Q - West")
    plt.savefig("Q_cliff.png")
plt.clf()
plt.close()

path = np.zeros((nrows, ncols, nact), dtype=np.float64)
x, y = go_to_start()
while(True):
    a = exploit(x, y, Q)
    print(x, y, a)
    x1, y1, state = move(x, y, a)
    if state == 1 or state == 2:
        print("Finished:" , state)
        break
    elif state == 0:
        x = x1
        y = y1
    if 0 <= x <= ncols - 1 and 0 <= y <= nrows - 1:
        path[y, x] = .2
plt.imshow(path)
plt.savefig("Q_path.png")
plt.clf()
plt.close()












