import numpy as np
from make_env import make_env
env = make_env('simple_adversary')

print('number of agents',env.n)
print('observation space',env.observation_space)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()
print(observation)

no_op = np.array([0,0.1,0.12,0.33,0.54])
action = [no_op,no_op,no_op]
obs_,reward,done,infor = env.step(action)
print(reward)
print(done)
