#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 19:04:30 2021

@author: Steven Lawrence
"""
import torch
from torch import nn
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename", default="input/demonstration.npy", help="the demonstration inputs")
parser.add_argument("--mode", default="train", choices=["train", "test", "demo"]) # , required=True
parser.add_argument("--model_dir", default="saved_models", help="where to save/restore the model")
parser.add_argument("--demo_length", default=20, help="number of demonstrations to do")

parser.add_argument("--max_episodes", type=int, default=3000, help="the number of training episodes")
parser.add_argument("--M", type=int, default=3000, help="the number of post demonstration examples") 


parser.add_argument("--batch_size", type=int, default=32, help="number of examples in batch")
parser.add_argument("--lr", type=float, default=0.0003, help="initial learning rate for adam SGD")
parser.add_argument("--w1", type=float, default=0.6, help="loss weight to improve action predictions") 
parser.add_argument("--w2", type=float, default=0.4, help="loss weight to imporve goal oriented task") 

parser.add_argument("--save_freq", type=int, default=100, help="save model every save_freq iterations, 0 to disable")
parser.add_argument("--print_freq", type=int, default=50, help="print current reward and loss every print_freq iterations, 0 to disable")

args = parser.parse_args()

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def get_shuffle_idx(num, batch_size):
  tmp = np.arange(num)
  np.random.shuffle(tmp)
  split_array = []
  cur = 0
  while num > batch_size:
    num -= batch_size
    if(num != 0):
      split_array.append(cur+batch_size)
      cur+=batch_size
  return np.split(tmp, split_array)