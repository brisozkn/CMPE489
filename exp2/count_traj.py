#!/usr/bin/env python
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from env import  GeneralEnv_Elfi
from agent import Agent

save_path = sys.argv[1]
taskID = sys.argv[2]
taskIdx = sys.argv[3]
reward_times = sys.argv[4]
gamma = sys.argv[5]
save_path = save_path+taskID+'/'+taskIdx+'/'

with open(os.getcwd()+'/test_parameters.pkl', 'rb') as f:
    pars = pkl.load(f)

"------------------------------------------------environment description------------------------------------------------------------------"
taskID = int(taskID)
envshape = pars[taskID]['env_shape']
rewpos = pars[taskID]['rew_position']
startstate = pars[taskID]['start_state']
stopstate = pars[taskID]['stop_state']
PathGreen = pars[taskID]['PathGreen']
PathBlue = pars[taskID]['PathBlue']
PathBlueViolet = pars[taskID]['PathBlueViolet']

def trial_test_generator(theta1, theta2, theta3, batch_size=1, random_state=None):
    data = []
    theta_env = [theta1]
    theta_goal = [float(reward_times)*theta2, float(reward_times)*theta3]
    # presentation test
    env_shape = envshape
    rew_pos = rewpos
    start_state = startstate
    stop_state = stopstate

    envir = GeneralEnv_Elfi(theta_env, theta_goal, env_shape, rew_pos, start_state, stop_state=stop_state)

    # train agent
    age = Agent(envir, temp=3, gamma=float(gamma))
    age.log = False
    for i in range(100000):
        age.do_iteration()

    # collect data
    age.softmax_temp = 0
    age.learning = False
    age.log = False
    age.clear()
    age.flag = 0
    data.append(envir.state)
    for i in range(100):
        age.do_iteration()
        if age.flag:
            data.append(envir.final_state[0])
            break
        data.append(envir.state)

    return data

"------------------------------------------------------------------------------------------------------------------------"

with open(save_path+"samples-"+gamma+".pkl", "rb") as f:
    theta_samples = pkl.load(f)
traj = []
# see 1000 paths
for sample in theta_samples:
    theta1_, theta2_, theta3_ = sample
    trj = trial_test_generator(theta1_, theta2_, theta3_)
    traj.append(trj)
# use mode
# import random
# from random import choice
# for i in range(300):
    # theta1_ = []
    # theta2_ = []
    # theta3_ = []
    # for sample_times in range(20):
        # sample = choice(theta_samples)
        # theta1_.append(sample[0])
        # theta2_.append(sample[1])
        # theta3_.append(sample[2])
    # theta1_ = max(theta1_)
    # theta2_ = max(theta2_)
    # theta3_ = max(theta3_)
    # trj = trial_test_generator(theta1_, theta2_, theta3_)
    # traj.append(trj)

clean_traj = []
clean_path = []
for pat in traj:
    clean_path = []
    clean_path.append(pat[0])
    for i in range(1, len(pat)):
        if pat[i] != pat[i-1]:
            clean_path.append(pat[i])
    clean_traj.append(clean_path)

count = {}
for pat in clean_traj:
    pat = tuple(pat)
    try:
        count[pat] += 1
    except KeyError:
        count[pat] = 1

with open(save_path+'count_reward'+reward_times+'gamma'+gamma+'.pkl', 'wb') as f:
    pkl.dump(count, f)

CountGreen = [0]*len(PathGreen)
CountBlue = [0]*len(PathBlue)
CountBlueViolet = [0]*len(PathBlueViolet)

for path in count.keys():
    path = list(path)
    num = count[tuple(path)]
    if path in PathGreen:
        idx = PathGreen.index(path)
        CountGreen[idx] += num
    elif path in PathBlue:
        idx = PathBlue.index(path)
        CountBlue[idx] += num
    elif path in PathBlueViolet:
        idx = PathBlueViolet.index(path)
        CountBlueViolet[idx] += num

with open(save_path+'count_green_reward'+reward_times+'gamma'+gamma+'.pkl', 'wb') as f:
    pkl.dump(CountGreen, f)

with open(save_path+'count_blue_reward'+reward_times + 'gamma' + gamma + '.pkl', 'wb') as f:
    pkl.dump(CountBlue, f)

with open(save_path+'count_violet_reward'+reward_times +'gamma' + gamma +'.pkl', 'wb') as f:
    pkl.dump(CountBlueViolet, f)