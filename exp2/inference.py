#!/usr/bin/env python
import elfi
import GPy
import sys
import os

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pickle as pkl
import random
from collections import Counter

from env import GeneralEnv_Elfi
from agent import Agent

save_path = sys.argv[1]
taskID = sys.argv[2]
taskIdx = sys.argv[3]
gamma = sys.argv[4]
save_path = save_path+taskID+'/'+taskIdx+'/'

with open(os.getcwd()+'/parameters.pkl', 'rb') as f:
    pars = pkl.load(f)

reward_weight = 0.1
"------------------------------------------------environment description------------------------------------------------------------------"
taskID = int(taskID)
obs = pars[taskID]['obs']
envshape1 = pars[taskID]['env_shape']
rewpos1 = pars[taskID]['rew_position']
theta1_first = pars[taskID]['theta1']
theta2_first = pars[taskID]['theta2']

"--------------------------------------------------------data generator---------------------------------------------------------------------"

def stimulus_one_generator(theta0, theta1, theta2, batch_size=1, random_state=None):
    data = []
    theta_env = [theta0]
    if theta1_first[0]:
        if theta2_first[0]:
            theta_goal = [theta1, theta2]
        else:
            theta_goal = [theta1]
    else:
        if theta2_first[0]:
            theta_goal = [theta2]
        else:
            theta_goal = []
    env_shape = envshape1
    rew_pos = rewpos1
    start_state = obs[0]
    stop_state = [16]
    envir = GeneralEnv_Elfi(theta_env, theta_goal, env_shape, rew_pos, start_state, stop_state=stop_state)

    # train agent
    age = Agent(envir, temp=3, gamma=float(gamma))
    age.log = False
    for i in range(100000):
        age.do_iteration()

    # collect data
    age.softmax_temp = 0.1
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

"------------------------------------------------------summary---------------------------------------------------------------"
def goa1_one_reached(trajectory, theta1_info):
    use, pos = theta1_info
    # trajectory = trajectory[idx]
    if use:
        count = Counter(trajectory)
        if pos in count:
            return 4*count[pos]
        else:
            return 0
    else:
        return 0

def goal_two_reached(trajectory, theta2_info):
    use, pos = theta2_info
    # trajectory = trajectory[idx]
    if use:
        count = Counter(trajectory)
        if pos in count:
            return 4*count[pos]
        else:
            return 0
    else:
        return 0

def count_grey_area(trajectory, env_info):
    grey_area = env_info[-1]
    # trajectory = trajectory[idx]
    if grey_area == []:
        return 0
    count = 0
    for pos in trajectory:
        if pos in grey_area:
            count += 1
    return reward_weight*count

def count_white_area(trajectory, env_info):
    grey_area = env_info[-1]
    # trajectory = trajectory[idx]
    if grey_area == []:
        return 0
    count = 0
    for pos in trajectory:
        if pos in grey_area:
            count += 1
    return (len(trajectory) - count)

def torch_grey(trajectory, env_info):
    if count_grey_area(trajectory, env_info) >0:
        return 1
    else:
        return 0
"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++inference ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
seed = np.random.randint(1000000)
np.random.seed(seed)
m = elfi.new_model()

theta0 = elfi.Prior(scipy.stats.norm, -0.5, 10, model=m)
theta1 = elfi.Prior(scipy.stats.norm, 0.5, 10, model=m)
theta2 = elfi.Prior(scipy.stats.norm, 0.5, 10, model=m)

Y1 = elfi.Simulator(stimulus_one_generator, theta0, theta1, theta2, observed=obs, model=m)
Sw_1 = elfi.Summary(count_white_area, Y1, envshape1)
Sg_1 = elfi.Summary(count_grey_area, Y1, envshape1)
Sy_1 = elfi.Summary(goa1_one_reached, Y1, theta1_first)
Sr_1 = elfi.Summary(goal_two_reached, Y1, theta2_first)
St_1 = elfi.Summary(torch_grey, Y1, envshape1)

d = elfi.Distance('euclidean', Sw_1, Sg_1, Sy_1, Sr_1, St_1, model=m)
# d = elfi.Distance('euclidean', Sw_1, Sg_1, Sy_1, Sr_1, model=m)
# log_d = elfi.Operation(np.log, d)

# create bolfi
bolfi = elfi.BOLFI(d, batch_size=1, initial_evidence=40, update_interval=20, bounds={'theta0':(-2, 0), 'theta1':(0, 2), 'theta2':(0, 2)}, seed=seed)

# start kernel part one
# kernel = GPy.kern.Exponential(input_dim=len(m.parameter_names), ARD=True)+GPy.kern.Bias(input_dim=len(m.parameter_names))
#kernel = GPy.kern.Matern52(input_dim=len(m.parameter_names), ARD=True)+GPy.kern.Bias(input_dim=len(m.parameter_names))
#target_mod = elfi.GPyRegression(m.parameter_names, bounds={'theta0':(-2, 0), 'theta1':(0, 2), 'theta2':(0, 2)}, kernel=kernel)
#bolfi = elfi.BOLFI(d, batch_size=1, initial_evidence=20, update_interval=10, bounds={'theta0':(-2, 0), 'theta1':(0, 2), 'theta2':(0, 2)}, target_model=target_mod, seed=seed)
# end kernel part one
# stop RL in 1000 steps, do_iteration loop

bolfi.fit(n_evidence=200)
np.savez(save_path+"independent-stimuli", X=bolfi.target_model.X, Y=bolfi.target_model.Y, params=bolfi.target_model._gp.param_array)

# target_model = elfi.GPyRegression(m.parameter_names, bounds={'theta0':(-1, 0), 'theta1':(0, 2), 'theta2':(0, 2)}, gp=m_load)
# bolfi = elfi.BOLFI(d, target_model=target_model)

# bolfi.state['n_evidence']=1
# bolfi.state['n_batches']=1

# bolfi.target_model._kernel_is_default=False

result_BOLFI = bolfi.sample(1000, info_freq=1000)
result_BOLFI.plot_pairs()
plt.savefig(save_path+'independent-stimulus_posterior.png')

idx = random.sample(range(2000), 1000)
theta_samples = [[result_BOLFI.samples['theta0'][i], result_BOLFI.samples['theta1'][i], result_BOLFI.samples['theta2'][i]] for i in idx]

with open(save_path+"samples-"+gamma+".pkl", 'wb') as f:
    pkl.dump(theta_samples, f)