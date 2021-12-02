#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:08:11 2021

@author: Steven Lawrence
"""
import torch
import numpy as np
from utils import args, get_shuffle_idx
import time
import os
import matplotlib.pyplot as plt

class BCO():
    def __init__(self, state_size, action_size, idm_state_size=None):
        # set initial values
        self.device        = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.state_dim     = state_size             # state dimension
        self.idm_state_dim = idm_state_size         # state dimension (agent specific)
        self.action_dim    = action_size            # action dimension
        self.lr            = args.lr                # model update learning rate
        self.max_episodes  = args.max_episodes      # maximum episode
        self.batch_size    = args.batch_size        # batch size
        self.alpha         = 0.006                  # alpha = | post_demo | / | pre_demo |
        self.M             = args.M                 # sample to update inverse dynamic model
        self.w1            = args.w1                # loss weight to improve action preditions
        self.w2            = args.w2                # loss weight to imporve goal oriented task

        self.state         = torch.tensor([self.state_dim]).unsqueeze(0)
        self.nstate        = torch.tensor([self.state_dim]).unsqueeze(0)
        self.action        = torch.tensor([self.action_dim]).unsqueeze(0)
        self.idm_state     = torch.tensor([self.idm_state_dim]).unsqueeze(0)
        self.idm_nstate    = torch.tensor([self.idm_state_dim]).unsqueeze(0)

        self.policy_model  = self.build_policy_model()
        self.idm_model     = self.build_idm_model()
        
        self.policy_model['model'].to(self.device)
        self.idm_model['model'].to(self.device)
        
        

    def build_policy_model(self):
        """buliding the policy model as two fully connected layers with leaky relu"""
        raise NotImplementedError

    def build_idm_model(self):
        """building the inverse dynamic model as two fully connnected layers with leaky relu"""
        raise NotImplementedError

    def load_demonstration(self):
        """Load demonstration from the file"""
        if args.input_filename is None or not os.path.isfile(args.input_filename):
          raise Exception("input filename does not exist")
    
        observations      = np.load(args.input_filename, allow_pickle=True)
        observations_next = observations[:,1]
        observations      = observations[:,0]
        
        obs      = np.array([o['observation']   for o in observations])
        ag       = np.array([o['achieved_goal'] for o in observations])
        g        = np.array([o['desired_goal']  for o in observations])
        obs_next = np.array([o['observation']   for o in observations_next])
        ag_next  = np.array([o['achieved_goal'] for o in observations_next])
        g_next   = np.array([o['desired_goal']  for o in observations_next])
        
        inputs   = {'obs':obs,      'ag':ag,      'g':g     }
        targets  = {'obs':obs_next, 'ag':ag_next, 'g':g_next}

        num_samples = obs.shape[0]
    
        return num_samples, inputs, targets

    def sample_demo(self):
        """sample demonstration"""
        sample_idx = range(self.demo_examples)
        sample_idx = np.random.choice(sample_idx, self.num_sample)
        
        S   = torch.FloatTensor([self.inputs['obs'][i]  for i in sample_idx])
        G   = torch.FloatTensor([self.inputs['g'][i]    for i in sample_idx])
        AG  = torch.FloatTensor([self.inputs['ag'][i]   for i in sample_idx])
        S   = torch.cat((S, G, AG), dim=1)
        
        nS  = torch.FloatTensor([self.targets['obs'][i] for i in sample_idx])
        nG  = torch.FloatTensor([self.targets['g'][i]   for i in sample_idx])
        nAG = torch.FloatTensor([self.targets['ag'][i]  for i in sample_idx])
        nS  = torch.cat((nS, nG, nAG), dim=1)
        
        return S, nS
    
    def eval_policy(self, state):
        """get the action by current state"""
        return self.policy_model['model'](state.to(self.device))

    def eval_idm(self, state, nstate):
        """get the action by inverse dynamic model from current state and next state"""
        if self.idm_state_dim != None:
            state, nstate = self.parse_idm_state(state, nstate)
        inputs = torch.cat((state,nstate),dim=1).to(self.device)
        return self.idm_model['model'](inputs)

    def pre_demonstration(self):
        """uniform sample action to generate (s_t, s_t+1) and action pairs"""
        raise NotImplementedError

    def post_demonstration(self):
        """using policy to generate (s_t, s_t+1) and action pairs"""
        raise NotImplementedError

    def eval_rwd_policy(self):
        """getting the reward by current policy model"""
        raise NotImplementedError
    
    def parse_idm_state(self):
        """for environments with seperate agent specific state space"""
        raise NotImplementedError
    
    def update_policy(self, state, action):
        """update policy model"""
        num  = len(state)
        idxs = get_shuffle_idx(num, self.batch_size)
        for idx in idxs:
            with torch.no_grad():
                batch_s = torch.stack([state[i]  for i in idx])
                batch_a = torch.stack([action[i] for i in idx])
                
            loss = self.get_policy_loss(batch_s, batch_a)
            self.policy_model['optimizer'].zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy_model['model'].parameters(), 0.25)
            self.policy_model['optimizer'].step()

    def update_idm(self, state, nstate, action):
        """update inverse dynamic model"""
        num  = len(state)
        idxs = get_shuffle_idx(num, self.batch_size)
        for idx in idxs:
            with torch.no_grad():
                batch_s  = torch.FloatTensor(np.array([state[i]  for i in idx]))
                batch_ns = torch.FloatTensor(np.array([nstate[i] for i in idx]))
                batch_a  = torch.FloatTensor(np.array([action[i] for i in idx]))
            loss = self.get_idm_loss(batch_s, batch_ns, batch_a)
            self.idm_model['optimizer'].zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.idm_model['model'].parameters(), 0.25)
            self.idm_model['optimizer'].step()

    def get_policy_loss(self, state, action):
        """get policy model loss"""
        pred = self.eval_policy(state[:,:28])
        goal = state[:,-6:-3]
        achieved_goal = state[:,-3:]
        
        action_loss = self.w1 * self.policy_model['loss'](pred, action) 
        goal_loss   = self.w2 * self.policy_model['loss'](achieved_goal, goal)
        return action_loss + goal_loss

    def get_idm_loss(self, state, nstate, action):
        """get inverse dynamic model loss"""
        pred = self.eval_idm(state, nstate)
        return self.idm_model['loss'](pred, action.to(self.device))
    
    
    def train(self):
        """training the policy model and inverse dynamic model by behavioral cloning"""
            
        print("\n[Training]")
        policy_losses = []
        idm_losses = []
        # pre demonstration to update inverse dynamic model
        S, nS, A = self.pre_demonstration()
        self.update_idm(S, nS, A)
        start = time.time()
        for i_episode in range(self.max_episodes):
          def should(freq):
            return freq > 0 and ((i_episode+1) % freq==0 or i_episode == self.max_episodes-1 )
    
          # update policy pi
          S, nS = self.sample_demo()
          A = self.eval_idm(S, nS)
          self.update_policy(S, A)
          policy_loss = self.get_policy_loss(S, A)
    
          # update inverse dynamic model
          S, nS, A = self.post_demonstration()
          self.update_idm(S, nS, A)
          idm_loss = self.get_idm_loss(torch.FloatTensor(S), 
                                       torch.FloatTensor(nS), 
                                       torch.FloatTensor(A))
         
          
          if should(args.print_freq):
            now = time.time()
            print('Episode: {:5d}, total reward: {:5.1f}, policy loss: {:8.6f}, idm loss: {:8.6f}, time: {:5.3f} sec/episode'.format((i_episode+1), self.eval_rwd_policy(), policy_loss, idm_loss, (now-start)/args.print_freq))
            start = now
    
          # saving model
          if should(args.save_freq):
            torch.save(self.policy_model['model'].state_dict(), args.model_dir + "/policy_model.pt")
            torch.save(self.idm_model['model'].state_dict(), args.model_dir + "/idm_model.pt")
            print('saving models: {}'.format(args.model_dir))
        
          policy_losses.append(policy_loss.detach().cpu().numpy())
          idm_losses.append(idm_loss.detach().cpu().numpy())

        plt.plot(np.array(policy_losses), label="Policy Loss", color="#47DBCD")
        plt.xlabel('Episode', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(frameon=False)
        plt.show()
        plt.plot(np.array(idm_losses), label="IDM Loss", color="#F5B14C")
        plt.xlabel('Episode', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend(frameon=False)
        plt.show()

    def load_policy_model(self):    
        state_dict = torch.load(args.model_dir + "/policy_model.pt")
        self.policy_model['model'].load_state_dict(state_dict)
        
    def test(self):
        self.load_demonstration()
        print('\n[Testing]\nFinal reward: {:5.1f}'.format(self.eval_rwd_policy()))

    def run(self):
        if not os.path.exists(args.model_dir):
          os.makedirs(args.model_dir)
    
        if args.mode == 'test':
            if args.model_dir is None:
                raise Exception("checkpoint required for test mode")
            self.test()
        
        if args.mode == 'demo':
            if args.model_dir is None:
                raise Exception("checkpoint required for test mode")
            self.load_policy_model()
            self.demo(args.demo_length)

        if args.mode == 'train':
          # read demonstration data
          self.demo_examples, self.inputs, self.targets = self.load_demonstration()
          self.num_sample = self.M
    
          self.train()
    
    
    
    
    
    