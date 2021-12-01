
import numpy as np
class MultiAgentReplayBuffer:
    def __init__(self,max_size,critic_dims,actor_dims,
            n_actions,n_agents,batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.actor_dims = actor_dims

        self.state_memory = np.zeros((self.mem_size,critic_dims))
        self.new_state_memory = np.zeros((self.mem_size,critic_dims))
        self.reward_memory = np.zeros((self.mem_size,n_agents))
        self.terminal_memory = np.zeros((self.mem_size,n_agents),dtype =bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory=[]
        
        for i in range(self.n_agents):
            self.actor_state_memory.append(
                    np.zeros((self.mem_size,self.actor_dims[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size,self.actor_dims[i]))
            )
            self.actor_action_memory.append(
                np.zeros((self.mem_size,self.n_actions))
            )
    def store_transition(self,raw_obs,state,action,reward,
            raw_obs_, state_,done):
        

