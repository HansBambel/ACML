import gym
import numpy as np
import matplotlib.pyplot as plt


gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
# env = gym.make("MountainCar-v0")
# print(env.observation_space.low)
# print(env.observation_space.high)
bins = 50

f1Bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0], bins)
f2Bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1], bins)
binSizes = (env.observation_space.high - env.observation_space.low) / bins
print(f1Bins)
print(f2Bins)
stateVisits = np.zeros((len(f1Bins), len(f2Bins), env.action_space.n))


def getIndex(obs):
    normedObs = (obs - env.observation_space.low) / binSizes
    return [int(normedObs[0])-1, int(normedObs[1])-1]


def train(qValues, GAMMA, ALPHA, episodes=50, save=False, backtracking=False):
    rewards = []
    for i in range(episodes+1):
        totalReward = 0
        observation = env.reset()
        i1, i2 = getIndex(observation)
        done = False
        timesteps = 0
        updateQ = []
        while not done:
            timesteps += 1
            # exploration vs exploitation
            # epsilon greedy maybe?
            # epsK = 1/(np.min(stateVisits[i1, i2]) + 1)
            # print(epsK)
            epsK = 0.1
            if np.random.random() < epsK:
                action = np.random.choice(len(qValues[i1, i2]))
            else:
                # softmax = np.exp(qValues[i1, i2])/np.sum(np.exp(qValues[i1, i2]))
                # action = np.random.choice(len(qValues[i1, i2]), p=softmax)
                action = np.argmax(qValues[i1, i2])
            # # print(qValues[i1, i2], softmax)

            # action = env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)
            totalReward += reward
            newi1, newi2 = getIndex(observation)

            # update Q-Value
            alpha = 1/(stateVisits[i1, i2, action]+1)
            alpha = max(alpha, ALPHA)
            qValues[i1, i2, action] += alpha * (reward + GAMMA * np.max(qValues[newi1, newi2]) - qValues[i1, i2, action])
            stateVisits[i1, i2, action] += 1
            updateQ.append([i1, i2, action])
            i1, i2 = newi1, newi2
            if done:
                updateQ.pop()

        if backtracking:
            # update q-values in reverse order --> faster convergence
            while len(updateQ) > 0:
                oldi1, oldi2, oldaction = updateQ.pop()
                alpha = 1/(stateVisits[oldi1, oldi2, oldaction]+1)
                alpha = max(alpha, ALPHA)
                qValues[oldi1, oldi2, oldaction] += alpha * (GAMMA * np.max(qValues[i1, i2]) - qValues[oldi1, oldi2, oldaction])
                stateVisits[oldi1, oldi2, oldaction] += 1
                i1, i2, action = oldi1, oldi2, oldaction
        if i%10 == 0:
            if backtracking:
                plotValues(qValues, i, save=True, folder='backtracking')
            else:
                plotValues(qValues, i, save=True, folder='noBacktracking')
        # save Q-Table
        if save:
            with open(f'qTable_{episodes}_episodes.npy', 'wb') as f:
                np.save(f, qValues)
        # save reward for later plotting
        rewards.append(totalReward)
        print(f"Episode {i:4d} finished after {timesteps} timesteps with reward {totalReward}.")
    print(f'Highest reward: {np.max(rewards)}')
    return rewards

def plotValues(qValues, episodes, save=False, folder='normal'):
    fig, ax = plt.subplots()
    ax.set_title(f'Value function of states using Q-Learning with {episodes} episodes')
    plt.imshow(np.max(qValues, axis=2))
    # plt.xlabel('Position')
    # plt.ylabel('Velocity')
    # ax.set_xticklabels(np.around(f1Bins, decimals=2))
    # ax.set_yticklabels(np.around(f2Bins, decimals=2))
    cbar = plt.colorbar()
    cbar.set_label('Value')
    if save:
        plt.savefig(f'fig/{folder}/Values after {episodes} episodes.png')
        plt.close()
    else:
        plt.show()


# Simulate with learned Q-Values
def runSimulation(qValues):
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
    print(f'Simulation ended with total reward: {totalReward}')


np.random.seed(42)
GAMMA = 0.9
# this alpha is actually a lowerbound
ALPHA = 0.03
episodes = 10000
qValues = np.zeros((len(f1Bins), len(f2Bins), env.action_space.n))
# with open('qTable_500_episodes_noBacktracking.npy', 'rb') as f:
#     qValues = np.load(f)
allRewards = train(qValues, GAMMA, ALPHA, episodes, save=True, backtracking=False)

plotValues(qValues, episodes)
fig, ax = plt.subplots()
ax.plot(allRewards)
ax.set_title('Rewards over time')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward of episode')
plt.show()

print(f'Q-Values Mean: {np.mean(qValues)}')
runSimulation(qValues)
# print(qValues)
