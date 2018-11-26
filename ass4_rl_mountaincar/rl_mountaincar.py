import gym
import numpy as np
import matplotlib.pyplot as plt


gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')
# env = gym.make("MountainCar-v0")
# print(env.observation_space.low)
# print(env.observation_space.high)



def getIndex(obs, binSizes):
    normedObs = (obs - env.observation_space.low) / binSizes
    return [int(normedObs[0])-1, int(normedObs[1])-1]


def train(qValues, binSizes, GAMMA, episodes=50, save=False, backtracking=False):
    rewards = []
    for i in range(episodes+1):
        totalReward = 0
        observation = env.reset()
        i1, i2 = getIndex(observation, binSizes)
        done = False
        timesteps = 0
        updateQ = []
        while not done:
            timesteps += 1
            action = np.argmax(qValues[i1, i2])

            observation, reward, done, info = env.step(action)
            totalReward += reward
            newi1, newi2 = getIndex(observation, binSizes)

            # update Q-Value
            qValues[i1, i2, action] = reward + GAMMA * np.max(qValues[newi1, newi2])
            updateQ.append([i1, i2, action])
            i1, i2 = newi1, newi2
            if done:
                updateQ.pop()

        if backtracking:
            # update q-values in reverse order --> faster convergence
            while len(updateQ) > 0:
                oldi1, oldi2, oldaction = updateQ.pop()
                qValues[oldi1, oldi2, oldaction] = reward + GAMMA * np.max(qValues[i1, i2])
                i1, i2, action = oldi1, oldi2, oldaction
        if i%10 == 0:
            if backtracking:
                plotValues(qValues, i, GAMMA, save=True, folder='backtracking')
            else:
                plotValues(qValues, i, GAMMA, save=True, folder='noBacktracking')
        # save Q-Table
        if save:
            with open(f'qTable_{episodes}_episodes.npy', 'wb') as f:
                np.save(f, qValues)
        # save reward for later plotting
        rewards.append(totalReward)
        print(f"Episode {i:4d} finished after {timesteps:5d} timesteps with reward {totalReward:5.0f}.")
    print(f'Highest reward in training: {np.max(rewards)}')
    return rewards

def plotValues(qValues, episodes, gamma, save=False, folder='normal'):
    fig, ax = plt.subplots()
    ax.set_title(f'Value function of states with {episodes} episodes. Gamma: {gamma}')
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
def runSimulation(qValues, binSizes):
    done = False
    observation = env.reset()
    totalReward = 0
    while not done:
        env.render()
        i1, i2 = getIndex(observation, binSizes)
        action = np.argmax(qValues[i1, i2])
        observation, reward, done, info = env.step(action)
        totalReward += reward
    env.close()
    print(f'Simulation ended with total reward: {totalReward}')


bins = 50
GAMMA = 0.995
episodes = 1000
binSizes = (env.observation_space.high - env.observation_space.low) / bins
qValues = np.zeros((bins, bins, env.action_space.n))
# with open('qTable_500_episodes_noBacktracking.npy', 'rb') as f:
#     qValues = np.load(f)
allRewards = train(qValues, binSizes, GAMMA, episodes, save=False, backtracking=True)

plotValues(qValues, episodes, GAMMA)
# fig, ax = plt.subplots()
# ax.plot(allRewards)
# ax.set_title('Rewards over time')
# ax.set_xlabel('Episode')
# ax.set_ylabel('Reward of episode')
plt.show()

print(f'Q-Values Mean: {np.mean(qValues)}')
runSimulation(qValues, binSizes)
# print(qValues)
