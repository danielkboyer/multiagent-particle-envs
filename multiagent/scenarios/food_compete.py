import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class SurvivalWorld(World):
    def __init__(self):
        super().__init__()
        self.step_number = 0
        self.day_length = 25
        self.day_number = 0
        self.number_of_days = 5
        self.done = False
    def step(self):
        World.step(self)

        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.health <= 0:
                agent.alive = False
                agent.color = np.array([0.85, 0.35, 0.35])
            agent.health -=1
            for landmark in self.landmarks:
                if landmark.alive and self.is_collision(agent,landmark) :
                    agent.health += 25
                    landmark.alive = False
                    landmark.color = np.array([0, 0, 0])
       

        self.step_number +=1
        if self.step_number % self.day_length == 0:
            self.day_number += 1
            #spawn food
            for i, landmark in enumerate(self.landmarks):
                if not landmark.alive:
                    landmark.name = 'landmark %d' % i
                    landmark.collide = True
                    landmark.movable = False
                    landmark.size = 0.04
                    landmark.alive = True
                    landmark.color = np.array([0.15, 0.65, 0.15])
                    landmark.state.p_pos = np.random.uniform(-1, +1, self.dim_p)
                    landmark.state.p_vel = np.zeros(self.dim_p)

        if self.day_number > self.number_of_days:
            self.done = True


    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

class Scenario(BaseScenario):

    def make_world(self):
        world = SurvivalWorld()
      
        # set any world properties first
        world.dim_c = 2
        num_agents = 10
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = 10
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.10
            #The health of the agent
            agent.health = 50.0
            agent.alive = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.04
            landmark.alive = True
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i in range(0, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.65, 0.15])
       
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.health = 50
            agent.alive = True
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        #RESET WORLD VARIABLES 
        world.step_number = 0
        world.day_number = 0
        world.done = False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        if agent.health <= 0:
            return -1

        reward = 1
        for agentBuddy in world.agents:
            if agentBuddy is agent:
                continue
            if agentBuddy.health > 0:
                reward += .5
            else:
                reward -= .5
        return reward

    def adversary_reward(self, agent, world):
        return -1
    def done(self,agent,world):
        return world.done

    def observation(self, agent, world):
        
        
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if entity.alive:
                entity_pos.append(entity.state.p_pos)
            else:
                entity_pos.append(entity.state.p_pos * 1000)
     
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            pos_health_list = list(other.state.p_pos)
            pos_health_list.append(other.health)
            other_pos.append(pos_health_list)

        
        obs =  np.concatenate( other_pos+ entity_pos)
        obs = np.append(obs,agent.health)
        return obs
    
