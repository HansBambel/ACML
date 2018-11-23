import gym
import numpy as np
import matplotlib.pyplot as plt


gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
slow = False
# env = gym.make("MountainCar-v0")
print(env.observation_space.low)
print(env.observation_space.high)
bins = 20

f1Bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], bins)
f2Bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], bins)
binSizes = (env.observation_space.high - env.observation_space.low) / bins
print(f1Bins)
print(f2Bins)
qValues = np.zeros((len(f1Bins), len(f2Bins), env.action_space.n))
stateVisits = np.zeros((len(f1Bins), len(f2Bins), env.action_space.n))
print(f'qValues.shape {qValues.shape}')
GAMMA = 0.9
ALPHA = 0.6

def getIndex(obs):
    normedObs = (obs - env.observation_space.low) / binSizes
    return [int(normedObs[0])-1, int(normedObs[1])-1]

episodes = 1000
for i in range(episodes+1):
    totalReward = 0
    observation = env.reset()
    i1, i2 = getIndex(observation)
    done = False
    timesteps = 0
    updateQ = []
    while not done:
        if slow:
            env.render()
        timesteps += 1
        # TODO exploration vs exploitation
        # epsilon greedy maybe?
        # epsK = 1/(np.min(stateVisits[i1, i2]) + 1)
        # print(epsK)
        epsK = 0.05
        if np.random.random() < epsK:
            action = np.random.choice(len(qValues[i1, i2]))
        else:
            action = np.argmax(qValues[i1, i2])
        # softmax = np.exp(qValues[i1, i2])/np.sum(np.exp(qValues[i1, i2]))
        # action = np.random.choice(len(qValues[i1, i2]), p=softmax)
        # # print(qValues[i1, i2], softmax)

        # action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        totalReward += reward
        newi1, newi2 = getIndex(observation)
        # update last Q-Value
        if done:
            alpha = 1/(stateVisits[i1, i2, action]+1)
            qValues[i1, i2, action] += alpha * (reward + GAMMA * np.max(qValues[newi1, newi2]) - qValues[i1, i2, action])
            stateVisits[i1, i2, action] += 1
        else:
            updateQ.append([i1, i2, action])
        i1, i2 = newi1, newi2

    # update q-values in reverse order --> faster convergence
    while len(updateQ) > 0:
        oldi1, oldi2, oldaction = updateQ.pop()
        alpha = 1/(stateVisits[oldi1, oldi2, oldaction]+1)
        qValues[oldi1, oldi2, oldaction] += alpha * (GAMMA * qValues[i1, i2, action] - qValues[oldi1, oldi2, oldaction])
        stateVisits[oldi1, oldi2, oldaction] += 1
        i1, i2, action = oldi1, oldi2, oldaction
    # if i%50 == 0:
    print(f"Episode {i:4d} finished after {timesteps} timesteps with reward {totalReward}.")

# Simulate with learned Q-Values
done = False
observation = env.reset()
totalReward = 0
while not done:
    env.render()
    i1, i2 = getIndex(observation)
    action = np.argmax(qValues[i1, i2])
    observation, reward, done, info = env.step(action)
    totalReward += reward
env.close()

# calc state value
# print(np.sum(qValues, axis=2))
# plt.heatmap(np.sum(qValues, axis=2), cbarlabel='Reward')
fig, ax = plt.subplots()
ax.set_title(f'Value function of states using Q-Learning with {episodes} episodes')
plt.imshow(np.max(qValues, axis=2))
plt.xlabel('Position')
plt.ylabel('Velocity')
# ax.set_xticklabels(np.around(f1Bins, decimals=2))
# ax.set_yticklabels(np.around(f2Bins, decimals=2))
cbar = plt.colorbar()
# cbar.set_label('Reward')
plt.show()
print(f'Simulation ended with total reward: {totalReward}')
