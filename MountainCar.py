import numpy as np
import matplotlib.pyplot as plt
import gym

#define the environment
env = gym.make("MountainCar-v0")

#Divide the continuous state space into a table of discrete state space regions and create a Q table
n_divisions = [20]*len(env.observation_space.high)
step_size = (env.observation_space.high - env.observation_space.low)/n_divisions
Q = np.zeros(n_divisions + [env.action_space.n])

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/step_size
    return tuple(discrete_state.astype(np.int))

#hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPISODES = 20000
EP_WINDOW = 500
EPSILON = 0.8

#set a constant epsilon decay, such that it decays completly before half the episodes
start_epsilon_decay = 1
end_epsilon_decay = EPISODES // 2
epsilon_decay_value = EPSILON/(end_epsilon_decay - start_epsilon_decay)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

#train the Q table
for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.uniform() > EPSILON:
            action = np.argmax(Q[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_next_q = np.max(Q[new_discrete_state])
            current_q = Q[discrete_state+(action,)]
            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_next_q )
            Q[discrete_state+(action,)] = new_q 
        elif new_state[0] >= env.goal_position:
            Q[discrete_state+(action,)] = 0
        discrete_state = new_discrete_state
        
        if episode % EP_WINDOW == 0:
            env.render()
    
    if end_epsilon_decay >= episode >= start_epsilon_decay:
        EPSILON -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    if episode % EP_WINDOW == 0:
        average_reward = sum(ep_rewards[-EP_WINDOW:])/len(ep_rewards[-EP_WINDOW:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-EP_WINDOW:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-EP_WINDOW:]))
        
        print(f"Episode:{episode} avg:{average_reward} min:{min(ep_rewards[-EP_WINDOW:])} max:{max(ep_rewards[-EP_WINDOW:])}")

env.close()

#Display the stats
plt.title(f"Statistics of rewards in a windows of {EP_WINDOW} episodes")
plt.xlabel("episodes")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend()
plt.show()

#final simulation with the learned Q table
print("Result")
env.reset()
done = False
while not done:
    action = np.argmax(Q[discrete_state])
    new_state, reward, done, _ = env.step(action)
    env.render()
    new_discrete_state = get_discrete_state(new_state) 
    discrete_state = new_discrete_state
    
env.close()