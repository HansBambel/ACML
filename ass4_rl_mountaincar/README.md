Python 3.6 is required.
To train an agent execute rl_mountaincar.py
In the file in line 102 the number of bins can be specified (default = 50)
In line 103 gamma can be specified (default = 0.995)
In line 104 the number of episodes to train the agent is specified (default = 1000)
In order to save a Q-Table (for e.g testing a simulation) change the parameter "save" to True when calling the train function. This will also save every 10th plot to the folder "fig/backtracking/".
In order to reuse a Q-Table specify in line 107 the previously saved file. and uncomment this and the following line.
