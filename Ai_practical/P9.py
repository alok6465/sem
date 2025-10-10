#501(prac9)----19/09/19----
#AIM: Implement Reinforcement Learning algorithm.
#REQUIREMENT: (T.npy) file
#(can be implemented by "td" and "adp")
#(we are calculating probable utility by Bellman theorem)

import numpy as np

#In the script I defined the function return_state_utility() which is an implementation of the Bellman equation.
#Using this function we are going to print the utility of the state (1,1)
#param v the state vector
#param T transition matrix
#param u utility vector
#param reward for that state
#param gamma discount factor (to which state it shud give more preference)
#return the utility of the state(probability of theagent to remain in that state.)
#product between the utility and the transition probability,
#then sum up the value for each action.
def return_state_utility(v, T, u, reward, gamma):
    #4 actions up,left, right,down
    #action array initially contain all 0's uptil it is calculated
    
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)

def main():
    #Starting state vector
    #The agent starts from (1, 1)
    v = np.array([[0.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 0.0, 
                   1.0, 0.0, 0.0, 0.0]])

    #Transition matrix loaded from file
    #(It is too big to write here)
    #transition matrix is a huge 12x12x4 matrix (12 starting states, 12 next states, 4 actions)
    #where most of the values are zeros
    T = np.load("T.npy") #"T.npy" is an external file which will be provided

    #Utility vector
    #after certain iterations what will be the agent's probability is utility of that state
    u = np.array([[0.812, 0.868, 0.918,   1.0,
                   0.762,   0.0, 0.660,  -1.0,
                   0.705, 0.655, 0.611, 0.388]])

    #Defining the reward for state (1,1)
    reward = -0.4
    #Assuming that the discount factor is equal to 1.0
    #preference to which state shud agent go desirable and non desirable state
    gamma = 1.0

    #Use the Bellman equation to find the utility of state (1,1)
    utility_11 = return_state_utility(v, T, u, reward, gamma)
    print("Utility of state (1,1): " + str(utility_11))

main()

#https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html

"""
OUTPUT:

Utility of state(1,1): 0.3456
"""
