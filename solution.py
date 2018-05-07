##Cecilia Ferrando
##10-703 Deep Reinforcement Learning (Spring 2018)
##HW1


import gym
import deeprl_hw1.lake_envs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from queue import PriorityQueue

from deeprl_hw1.rl import *




def reform_pol(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    return str_policy


##CHOOSE GRID DIMENSION
#env = gym.make('Deterministic-4x4-FrozenLake-v0')
env = gym.make('Deterministic-8x8-FrozenLake-v0')
#print('I made the environment')
#print(env.P)


gamma = .9
action_names = {deeprl_hw1.lake_envs.LEFT: 'L', deeprl_hw1.lake_envs.RIGHT: 'R', deeprl_hw1.lake_envs.DOWN: 'D', deeprl_hw1.lake_envs.UP: 'U'}

dim = int(np.sqrt(env.nS))

####EXERCISE 2.I

##2.I.a

start = time.time()
print('I am here and computing')
print(env)
print(gamma)
#check
policy, valueFunction, policyIter, valueIter = policy_iteration_async_ordered(env,gamma,max_iterations=int(1e3), tol=1e-3)
print('I am still here and have computed')
runTime = str(time.time()-start)
print('Run time is %s seconds' %runTime)
print('%s policy improvement steps' %policyIter)
print('%s policy evaluation steps' %valueIter)

##2.I.b

print('2.I.b')
print('OPTIMAL POLICY')
print(dim)
for i in range(dim):
    print(reform_pol(policy,action_names)[dim*i:dim*(i+1)])

##2.I.c

print('2.I.c')
print('This policy have value-function: ')
for i in range(dim):
    print(valueFunction[dim*i:dim*(i+1)])
valFunRes = np.reshape(valueFunction,(dim,dim))
flipped = np.flipud(valFunRes)
fig = plt.pcolor(flipped,cmap=cm.coolwarm) 
print('here are the plots for part c')
plt.colorbar(fig)
plt.show()

##2.I.d

print('2.I.d')
startTime = time.time()
V, valueIters = value_iteration_sync(env,gamma)
elapsedTime = str(time.time() - startTime)
print('This is the optimal value function, directly from value iteration: ')

for i in range(dim):
    print(V[dim*i:dim*(i+1)])
    
print('number of iterations required is ')
print(valueIters)

##2.I.e

print('2.I.e')
valFunRes = np.reshape(V,(dim,dim))
flipped = np.flipud(valFunRes)
fig = plt.pcolor(flipped,cmap=cm.coolwarm)
plt.colorbar(fig)
plt.show()

##2.I.g

print('2.I.g')
print('I obtained this policy vector')
optimalPolicy = value_function_to_policy(env, gamma, V)
policyVector = reform_pol(optimalPolicy, action_names)
policyVector = np.reshape(policyVector, (dim,dim))
print(policyVector)
print('')

##2.I.h/i

#policy async ordered
print('2.I.h and 2.I.i')
start = time.time()
policy, valueFunction, policyIter, valueIter = policy_iteration_async_ordered(env,gamma,max_iterations=int(1e3), tol=1e-3)
runTime = str(time.time()-start)
print('2.I.i - async ordered')
print('Run time is %s seconds' %runTime)
print('%s policy improvement steps' %policyIter)
print('%s policy evaluation steps' %valueIter)

#policy async random permutation
start = time.time()
policy, valueFunction, policyIter, valueIter = policy_iteration_async_randperm(env,gamma,max_iterations=int(1e3), tol=1e-3)
runTime = str(time.time()-start)
print('2.I.i - async randperm')
print('Run time is %s seconds' %runTime)
print('%s policy improvement steps' %policyIter)
print('%s policy evaluation steps' %valueIter)

#value async ordered
start = time.time()
valueFunction, iter = value_iteration_async_ordered(env,gamma,max_iterations=int(1e3), tol=1e-3)
runTime = str(time.time()-start)
print('2.I.i - value async ordered')
print('Run time is %s seconds' %runTime)
print('%s iterations' %iter)
print('%s valueFunction' %valueFunction)

#value async random permutation
start = time.time()
valueFunction, iter  = value_iteration_async_randperm(env,gamma,max_iterations=int(1e3), tol=1e-3)
runTime = str(time.time()-start)
print('2.I.i - value async randperm')
print('Run time is %s seconds' %runTime)
print('%s iterations' %iter)
print('%s valueFunction' %valueFunction)


##2.I.j

print('2.I.j')
observation = env.reset()
reward = 0
observation,myReward,isTerminal,placeholder = env.step(optimalPolicy[observation])
nSteps = 0

reward += myReward   #cumulate reward

while isTerminal==False:
    if nSteps==1:
        print('Start reward', reward)
    nSteps += 1
    observation,myReward,isTerminal,placeholder = env.step(optimalPolicy[observation])
    #accumulating reward
    reward += (gamma**nSteps)*myReward
        
print('The cumulative discounted reward is %s,' %reward)


####EXERCISE 2.II


##2.II.a

print('2.II.a')

startTime = time.time()
V, valueIters = value_iteration_sync(env,gamma)
elapsedTime = str(time.time()-startTime)
print('Optimal value function is:')

for i in range(dim):
    print(V[dim*i : dim*(i+1)])
print('Run time is %s seconds' %elapsedTime)
print('Number of required iterations is %s' %valueIters)
print('\n')

##2.II.b

print('2.II.a')

#value function
valFunRes = np.reshape(V,(dim,dim))
#flip it upside down and plot
fig = plt.pcolor(np.flipud(valFunRes),cmap=cm.coolwarm)
plt.colorbar(fig)
plt.show()

print('The value function is different from the one in deterministic setting')

##2.II.c

print('2.II.c')
print('Converted value function to optimal policy, yields:')

optimalPolicy = value_function_to_policy(env, gamma, V)
#Convert to print the policy in human readable format:
policyVector = reform_pol(optimalPolicy, action_names)
policyVector = np.reshape(policyVector, (dim,dim))

print(policyVector)


##2.II.j

rewards = np.zeros(100)

for sample in range(100):   #take a 100 large sample
    
    observation = env.reset()
    #print env.reset()
    observation,myReward,isTerminal,placeholder=env.step(optimalPolicy[observation])
    numStep = 0
    rewards[sample] += (gamma**numStep) * myReward
    
    while isTerminal==False:
        if nSteps==1:
            print('Start reward', reward)
        numStep += 1
        observation,myReward,isTerminal,placeholder=env.step(optimalPolicy[observation])
        rewards[sample] += (gamma**numStep)*myReward
        
meanReward = np.mean(rewards)

print('The average cumulative discounted reward is %s' %meanReward)

#### EXERCISE 2.III

# start = time.time()
# policy, valueFunction, policyIter, valueIter = policy_iteration_async_custom(env,gamma,max_iterations=int(1e3), tol=1e-3)
# runTime = str(time.time()-start)
# print('2.I.i - async custom')
# print('Run time is %s seconds' %runTime)
# print('%s policy improvement steps' %policyIter)
# print('%s policy evaluation steps' %valueIter)

# start = time.time()
# valueFunction, iter  = value_iteration_async_custom(env,gamma,max_iterations=int(1e3), tol=1e-3)
# runTime = str(time.time()-start)
# print('2.I.i - value async custom')
# print('Run time is %s seconds' %runTime)
# print('%s iterations' %iter)
# print('%s valueFunction' %valueFunction)