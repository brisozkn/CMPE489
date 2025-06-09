#!/usr/bin/env python
import pandas as pd
import pickle as pkl
import sys
import os

save_path = sys.argv[1]
taskID = sys.argv[2]
taskIDX = sys.argv[3]
reward_times = sys.argv[4]
gamma = sys.argv[5]
AgentType = 'Qlearning'
save_path = save_path+taskID+'/'+taskIDX+'/'
group = 'Parallel'

try:
    model_data = pd.read_csv(os.getcwd()+'/model_data_reward'+reward_times+'gamma'+gamma+'.csv')
except FileNotFoundError:
    model_data = pd.DataFrame(columns=["TaskID", "TaskIndex", "SliderGreen", "SliderBlue", "SliderViolet", "Group", 'AgentType', 'BerryMultiplier'])

with open(save_path+'count_green_reward'+reward_times+'gamma'+gamma+'.pkl', 'rb') as f:
    count_green = pkl.load(f)
with open(save_path+'count_blue_reward'+reward_times+'gamma'+gamma+'.pkl', 'rb') as f:
    count_blue = pkl.load(f)
with open(save_path+'count_violet_reward'+reward_times+'gamma'+gamma+'.pkl', 'rb') as f:
    count_violet = pkl.load(f)

SliderGreen = sum(count_green)
SliderBlue = sum(count_blue)
SliderViolet = sum(count_violet)
Total = SliderGreen + SliderBlue + SliderViolet

if Total != 0:
    SliderGreen = SliderGreen/Total
    SliderBlue = SliderBlue/Total
    SliderViolet = SliderViolet/Total
else:
    SliderGreen = -1
    SliderBlue = -1
    SliderViolet = -1

taskID = int(taskID)
taskIDX = int(taskIDX)

model_data.loc[model_data.shape[0]] = [taskID, taskIDX, SliderGreen, SliderBlue, SliderViolet, group, AgentType, reward_times]

model_data.to_csv(os.getcwd()+'/model_data_reward'+reward_times+'gamma'+gamma+'.csv', columns=["TaskID", "TaskIndex", "SliderGreen", "SliderBlue", "SliderViolet", "Group", "AgentType", "BerryMultiplier"], index=False)