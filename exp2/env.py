import csv
import numpy as np

class Environment_ELFI():
    def __init__(self, theta1, theta2, max_row, max_col, size=1):
        self.actions = ["left", "right", "up", "down", "stay"]
        self.transition_matrix = self.build_transition_matrix(max_row, max_col)
        self.reward_map = self.build_reward_map(theta1, theta2, max_row, max_col)
        self.final_state = np.argmax(self.reward_map)
        self.clear()

    def build_transition_matrix(self, max_row, max_col):
        states = list(range(0, max_row*max_col))
        trans = dict()
        for item in states:
            trans[(item, self.actions[-1])] = item
            if (item % max_col) > 0:
                trans[(item, self.actions[0])] = item - 1
            if (item + 1) % max_col > 0:
                trans[(item, self.actions[1])] = item + 1
            if int(item/max_col) > 0:
                trans[(item, self.actions[2])] = item - max_col
            if int(item/max_col) < (max_row - 1):
                trans[(item, self.actions[3])] = item + max_col
        return trans

    def build_reward_map(self, theta1, theta2, max_row, max_col, size=1):
        map = np.zeros([max_row, max_col]) - 0.1
        map[int((max_row-size)/2):int((max_row+size)/2), int((max_col-size)/2):int((max_col+size)/2)] = theta1
        map[int((max_row+size)/2)-1, int((max_col+size)/2)] = theta2
        return np.reshape(map, (-1))

    def clear(self):
        self.state = 0
        self.reward = 0
        self.terminal_state = 0

    def step(self, action):
        transit = self.transition_matrix
        try:
            self.state = transit[(self.state, action)]
        except KeyError:
            self.state = self.state

        self.reward = self.reward_map[self.state]
        if self.state == self.final_state:
            self.terminal_state = 1

class GeneralEnv_Elfi():
    def __init__(self, theta_env, theta_goal, env_shape, rew_pos, start_state, stop_state, adding_terminate=True, final_rew=1):
        """
        :param theta_env: the environment costs of different states. The length of theta_env stands for the number of
        different envs
        :param theta_goal: The reward of goals. The length of theta_goal stands for the number of goals.
        :param env_shape: env_shape[0] stands for the overall environment shape. env_shape[1][0] is the starting position
        of theta_env[0], env_shape[1][1] is the end position of theta_env[1], ...
        :param rew_pos: similar to env_shape
        """
        self.actions = ["left", "right", "up", "down"]
        self.theta_env = theta_env
        self.theta_goal = theta_goal
        self.env_shape = env_shape[0]
        self.reward_pos = rew_pos
        self.start_state = start_state
        self.final_state = stop_state
        self.adding_terminate = adding_terminate
        self.final_state_reward = final_rew

        self.env_map = self.build_env_map(theta_env, env_shape, rew_pos) # np
        self.reward_map = self.build_reward_map(theta_goal, rew_pos) # np
        self.transition_matrix = self.build_transition_matrix(env_shape[0], rew_pos)
        # self.final_state = np.argmax(self.reward_map)
        self.goal_reached = [0]*len(rew_pos)
        self.clear()

    def build_transition_matrix(self, env_shape, rew_pos):
        max_row = env_shape[0]
        max_col = env_shape[1]
        states = list(range(0, max_row*max_col))
        trans = dict()
        for item in states:
            #trans[(item, self.actions[-1])] = item
            if (item % max_col) > 0:
                trans[(item, self.actions[0])] = item - 1
            if (item + 1) % max_col > 0:
                trans[(item, self.actions[1])] = item + 1
            if int(item/max_col) > 0:
                trans[(item, self.actions[2])] = item - max_col
            if int(item/max_col) < (max_row - 1):
                trans[(item, self.actions[3])] = item + max_col

        # states = states + [max_row * max_col]
        if self.adding_terminate:
            for row in range(1, max_row + 1):
                item = row * max_col - 1
                trans[(item, self.actions[1])] = max_row * max_col

        """
        for i in range(len(rew_pos)):
            trans_state = self.env_dict[rew_pos[i][0]]
            for act in self.actions:
                try:
                    trans[(trans_state, act)] = trans[(rew_pos[i][0], act)]
                except KeyError:
                    continue
            trans[(trans_state, self.actions[-1])] = trans_state
        """
        return trans

    def build_env_map(self, theta_env, env_shape, rew_pos):
        if self.adding_terminate:
            length = env_shape[0][0]*env_shape[0][1] + 1 #real states, terminate state
        else:
            length = env_shape[0][0]*env_shape[0][1]
        map = np.zeros(length) - 0.1 # white states punishment 0.1
        temp_count = 1
        if theta_env == []:
            return np.reshape(map, (-1))
        for theta in theta_env:
            for pos in env_shape[temp_count]:
                    map[pos] = theta # gray states punishment
            temp_count += 1
        if self.adding_terminate:
            for pos in self.final_state:
                map[pos] += self.final_state_reward

        """
        temp_count = 1
        self.env_dict = dict()
        for pos in rew_pos:
            map[env_shape[0][0]*env_shape[0][1] + temp_count] = map[pos]
            self.env_dict[pos[0]] = env_shape[0][0]*env_shape[0][1] + temp_count
            temp_count += 1
        """
        return np.reshape(map, (-1))


    def build_reward_map(self, theta, rew_pos):
        map = np.zeros(np.shape(self.env_map))
        # map = np.zeros(env_shape[0]*env_shape[1] + 1)
        for i in range(len(theta)):
            pos = rew_pos[i]
            map[pos] = theta[i]
            # for pos in rew_pos[i]:
            #     map[pos] = theta[i]
        return np.reshape(map, (-1))

    def clear(self):
        self.state = self.start_state
        self.reward = 0
        self.terminal_state = 0
        # self.reward_map = self.build_reward_map(theta=self.theta_goal, env_shape=self.env_shape, rew_pos=self.reward_pos)
        self.count_arrival = [0 for i in range(len(self.reward_map))]
        self.goal_reached = [0]*len(self.reward_pos)

    def step(self, action):
        transit = self.transition_matrix
        try:
            self.state = transit[(self.state, action)]
        except KeyError:
            self.state = self.state

        if (self.reward_map[self.state] != 0) and (self.count_arrival[self.state] == 0):
            self.reward = self.reward_map[self.state] + self.env_map[self.state]
            self.count_arrival[self.state] = 1
            idx = self.reward_pos.index(self.state)
            self.goal_reached[idx] = 1
        else:
            self.reward = self.env_map[self.state]


        if self.state in self.final_state:
            self.terminal_state = 1