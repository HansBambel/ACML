import gym
import numpy as np


slow = False
env = gym.make("MountainCar-v0")
print(env.observation_space.low)
print(env.observation_space.high)
bins = 20

f1Bins = np.linspace(env.observation_space.low[0], env.observation_space.high[0])
f2Bins = np.linspace(env.observation_space.low[1], env.observation_space.high[1])
print(f1Bins)
print(f2Bins)
numDiscreteStates = len(f1Bins*f2Bins)
qValues = np.zeros((len(f1Bins), len(f2Bins), env.action_space.n))


for _ in range(10):
    observation = env.reset()
    done = False
    timesteps = 0
    while not done:
        if slow:
            env.render()
        # TODO exploration vs exploitation
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        timesteps += 1
        if slow:
            print(observation)
        if slow:
            print(reward)
        if slow:
            print(done)
    print("Episode finished after ", timesteps, "timesteps.")
