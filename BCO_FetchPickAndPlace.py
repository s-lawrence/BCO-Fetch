#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:39:42 2021

@author: Steven Lawrence
"""
import torch
from torch import nn
import numpy as np
from BCO import BCO
from utils import init_weights
import gym

class BCO_FetchPickAndPlace(BCO):
    def __init__(self, state_size, action_size, idm_state_size):
        BCO.__init__(self, state_size, action_size, idm_state_size)
        self.env = gym.make('FetchPickAndPlace-v1')
        
    def build_policy_model(self):
        policy_model = {}
        policy_model['input'] = self.state
        policy_model['model'] = nn.Sequential(
            nn.Linear(policy_model['input'], 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.action_dim)
        ).apply(init_weights)
        policy_model['loss'] = nn.MSELoss()
        policy_model['optimizer'] = torch.optim.Adam(policy_model['model'].parameters(), lr = self.lr) 
        
        return policy_model
    
    def build_idm_model(self):
        idm_model = {}
        idm_model['input'] = torch.cat([self.idm_state + self.idm_nstate], 1)
        idm_model['model'] = nn.Sequential(
            nn.Linear(idm_model['input'], 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, self.action_dim)
        ).apply(init_weights)
        idm_model['loss'] = nn.MSELoss()
        idm_model['optimizer'] = torch.optim.Adam(idm_model['model'].parameters(), lr = self.lr) 
        
        return idm_model

    def pre_demonstration(self):
        """uniform sample action to generate (s_t, s_t+1) and action pairs"""
        terminal = True
        States = []
        Nstates = []
        Actions = []
    
        for i in range(int(round(self.M / self.alpha))):
            if terminal:
                observation = self.env.reset()
                state = observation['observation']
                g     = observation['desired_goal']
                ag    = observation['achieved_goal']
    
            prev_s  = state
            prev_g  = g
            prev_ag = ag
        
            a = self.env.action_space.sample()
        
            observation, _, terminal, _ = self.env.step(a)
            state = observation['observation']
            g     = observation['desired_goal']
            ag    = observation['achieved_goal']
          
            prev_s_g = np.concatenate([prev_s, prev_g, prev_ag], axis=0)          
            state_g  = np.concatenate([state, g, ag], axis=0)
          
            States.append(prev_s_g)
            Nstates.append(state_g)
            Actions.append(a)
        
            if i and (i+1) % 10000 == 0:
                print("Collecting idm training data ", i+1)
        
        return States, Nstates, Actions

    def post_demonstration(self):
        """using policy to generate (s_t, s_t+1) and action pairs"""
        terminal = True
        States = []
        Nstates = []
        Actions = []
    
        for i in range(self.M):
            if terminal:
                observation = self.env.reset()
                state = observation['observation']
                g     = observation['desired_goal']
                ag    = observation['achieved_goal']
    
            prev_s  = state
            prev_g  = g
            prev_ag = ag
            
            state   = torch.from_numpy(state)
            g       = torch.from_numpy(g)
            state = torch.cat((state, g), 0).float().to(self.device)
      
            a = np.reshape(self.eval_policy(state).detach().cpu().numpy(), [-1])
            
            observation, _, terminal, _ = self.env.step(a)
            state = observation['observation']
            g     = observation['desired_goal']
            ag    = observation['achieved_goal']
            
            prev_s_g = np.concatenate([prev_s, prev_g, prev_ag], axis=0)          
            state_g  = np.concatenate([state, g, ag], axis=0)
      
            States.append(prev_s_g)
            Nstates.append(state_g)
            Actions.append(a)
    
        return States, Nstates, Actions

    def eval_rwd_policy(self):
        """getting the reward by current policy model"""
        terminal = False
        total_reward = 0
        observation = self.env.reset()
        state = observation['observation']
        g     = observation['desired_goal']
        ag    = observation['achieved_goal']
    
        while not terminal:
            state = torch.from_numpy(np.concatenate([state,g], axis=0).reshape([-1,self.state_dim])).float()
            a = np.reshape(self.eval_policy(state).cpu().detach().numpy(), [-1])
            observation, reward, terminal, _ = self.env.step(a)
            state = observation['observation']
            g     = observation['desired_goal']
            ag    = observation['achieved_goal']
            total_reward += reward
    
        return total_reward
    
    def parse_agent_specific_state(self, state):
        gripper_pos = state[:,:3]
        gripper_state = state[:,9:11]
        grip_and_gripper_vel = state[:,20:25]
    
        return torch.from_numpy(np.concatenate([gripper_pos, gripper_state, grip_and_gripper_vel], axis=1))
    
    def parse_idm_state(self, state, nstate):
        state  = self.parse_agent_specific_state(state)
        nstate = self.parse_agent_specific_state(nstate)
        return state, nstate
    
    def demo(self, demo_length):
        success = 0
        for i in range(demo_length):
            observation = self.env.reset()
            # start to do the demo
            state = observation['observation']
            g     = observation['desired_goal']
            for t in range(self.env._max_episode_steps):
                self.env.render()
                state = torch.from_numpy(np.concatenate([state,g], axis=0).reshape([-1,self.state_dim])).float()
                a     = np.reshape(self.eval_policy(state).cpu().detach().numpy(), [-1])
                observation, reward, terminal, info = self.env.step(a)
                state = observation['observation']
                g     = observation['desired_goal']
            print('the episode is: {}, is success: {}'.format(i, info['is_success']))
            success += info['is_success']
        print("Success rate: {}%".format(success/demo_length*100))
    
if __name__ == "__main__":
  bco = BCO_FetchPickAndPlace(28, 4, 10)
  bco.run()