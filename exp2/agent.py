from env import Environment_ELFI
import math
import random
from operator import itemgetter
import numpy as np


class Agent():
    def __init__(self, env, temp, gamma):
        self.env = env
        self.actions = ["left", "right", "up", "down"]
        self.softmax_temp = temp
        self.rand_value = temp
        self.alpha = 0.1
        self.gamma = gamma
        self.learning = True
        self.step_count = 0
        self.log = False
        self.write_to = "test.txt"
        self.counts = []

        self.flag = 0

        self.state = env.state
        self.goal_reached = tuple(env.goal_reached)
        self.max_step = 100
        """
        self.q = {}
        self.q[self.state] = {}F
        for a in self.actions:
            self.q[self.state][a] = 0
        """
        self.q = {}
        self.q[self.state] = {}
        self.q[self.state][self.goal_reached] = {}
        for a in self.actions:
            self.q[self.state][self.goal_reached][a] = 0
        self.clear()

    def clear(self):
        self.env.clear()
        self.action = None
        self.previous_state = None
        self.state = self.env.state
        self.reward = 0
        self.step_count = 0
        self.goal_reached = tuple(self.env.goal_reached)
        self.previous_goal_reached = None

    def set_state(self):
        self.previous_state = self.state
        self.state = self.env.state
        """
        if self.state not in self.q:
            self.q[self.state] = {}
            for a in self.actions:
                self.q[self.state][a] = 0
        """

        # self.previous_count = self.count_reward
        self.previous_goal_reached = self.goal_reached
        self.goal_reached = tuple(self.env.goal_reached)

        if self.previous_state not in self.q:
            self.q[self.previous_state] = {}
            self.q[self.previous_state][self.previous_goal_reached] = {}
            for a in self.actions:
                self.q[self.previous_state][self.previous_goal_reached][a] = 0
        if self.previous_goal_reached not in self.q[self.previous_state]:
            self.q[self.previous_state][self.previous_goal_reached] = {}
            for a in self.actions:
                self.q[self.previous_state][self.previous_goal_reached][a] = 0

        if self.state not in self.q:
            self.q[self.state] = {}
            self.q[self.state][self.goal_reached] = {}
            for a in self.actions:
                self.q[self.state][self.goal_reached][a] = 0
        if self.goal_reached not in self.q[self.state]:
            self.q[self.state][self.goal_reached] = {}
            for a in self.actions:
                self.q[self.state][self.goal_reached][a] = 0

    def do_iteration(self, debug=False):
        self.set_state()
        self.choose_action_softmax()
        # self.choose_action()
        self.update_q_learning()
        self.env.step(self.action)
        self.step_count += 1

        self.reward = self.env.reward

        # if self.reward > 0:
        #     self.count_reward += [self.state]

        if self.log:
            with open(self.write_to, "a+") as f:
                print(self.state, file=f)

        if self.env.terminal_state or self.step_count >= self.max_step:
            # write the self.step_count to a log
            # zss, to remove no ops
            # self.reward = self.reward - self.step_count*(0.1)
            if self.log:
                self.counts.append(self.step_count)
                with open(self.write_to, "a+") as f:
                    print("task finished", file=f)

            self.update_q_td()
            # zss, a flag for the end of a term
            if self.step_count < self.max_step:
                self.flag = 1
            self.clear()

    def calculate_max_q_value(self):
        # TODO:
        # i = 0
        # if 7 in self.count_reward:
        #     i += 1
        # if 11 in self.count_reward:
        #     i += 2
        return max(self.q[self.state][self.goal_reached].items(), key=itemgetter(1))
        # return max(self.q[self.state].items(), key=itemgetter(1))

    def update_q_learning(self, debug=False):
        if self.previous_action != None and self.learning:
            try:
                previous_q = self.q[self.previous_state][self.previous_goal_reached][self.previous_action]
            except KeyError:
                if self.previous_state not in self.q:
                    self.q[self.previous_state] = {}
                if self.previous_goal_reached not in self.q[self.previous_state]:
                    self.q[self.previous_state][self.previous_goal_reached] = {}
                for a in self.actions:
                    self.q[self.previous_state][self.previous_goal_reached][a] = 0
                previous_q = 0
            next_q = self.calculate_max_q_value()[1]
            self.q[self.previous_state][self.previous_goal_reached][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    def update_q_td(self):
        # i = 0
        # if 7 in self.count_reward:
        #     i += 1
        # if 11 in self.count_reward:
        #     i += 2
        if self.learning and self.action != None:
            previous_q = self.q[self.state][self.goal_reached][self.action]
            self.q[self.state][self.goal_reached][self.action] = \
                previous_q + self.alpha * (self.reward - previous_q)

    # def update_q_sarsa(self):
    #     if self.learning and self.previous_action != None:
    #         previous_q = self.q[self.previous_state][self.previous_action]
    #         next_q = self.q[self.state][self.action]
    #         self.q[self.previous_state][self.previous_action] = \
    #             previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    # Used by softmax
    def weighted_random(self, weights):
        number = random.random() * sum(weights.values())
        for k, v in weights.items():
            if number < v:
                break
            number -= v
        return k

    def random_max_action(self):
        p = {}
        for a in self.q[self.state][self.goal_reached].keys():
            if self.softmax_temp != 0:
                try:
                    p[a] = math.exp(self.q[self.state][self.goal_reached][a] / self.softmax_temp)
                except OverflowError:
                    p = {}
                    for a in self.q[self.state][self.goal_reached].keys():
                        p[a] = self.q[self.state][self.goal_reached][a]
            else:
                for a in self.q[self.state][self.goal_reached].keys():
                    p[a] = self.q[self.state][self.goal_reached][a]
        s = sum(p.values())
        for a in p.keys():
            p[a] = p[a] / s
        max_p = max(p.items(), key=itemgetter(1))[-1]
        possible_actions = []
        for a in p.keys():
            # print(p[a])
            if p[a] >= max_p:
                possible_actions.append(a)
        return random.choice(possible_actions)

    def choose_action_softmax(self):
        self.previous_action = self.action
        if self.softmax_temp == 0:
            self.action = self.random_max_action()
            # self.action = max(self.q[self.state][self.goal_reached].items(), key=itemgetter(1))[0]
            return
        p = {}
        for a in self.q[self.state][self.goal_reached].keys():
            try:
                p[a] = math.exp(self.q[self.state][self.goal_reached][a] / self.softmax_temp)
            except OverflowError:
                self.action = self.random_max_action()
                # self.action = max(self.q[self.state][self.goal_reached].items(), key=itemgetter(1))[0]
                return
        s = sum(p.values())
        if s != 0:
            p = {k: v / s for k, v in p.items()}
            self.action = self.weighted_random(p)
        else:
            self.action = np.random.choice(list(p.keys()))

    def choose_action(self):
        self.previous_action = self.action
        poss = random.random()
        if poss < self.rand_value:
            self.action = random.choice(self.actions)
        else:
            self.action = max(self.q[self.state].items(), key=itemgetter(1))[0]

    def softmax_values(self, rounding=2):
        if self.softmax_temp == 0:
            p = {}
            best = max(self.q[self.state].items(), key=itemgetter(1))[0]
            for a in self.q[self.state].keys():
                if a == best:
                    p[a] = 1.0
                else:
                    p[a] = 0.0
            return p

        p = {}
        for a in self.q[self.state].keys():
            p[a] = math.exp(self.q[self.state][a] / self.softmax_temp)

        # Normalize
        s = sum(p.values())
        if s != 0:
            p = {k: round(v / s, rounding) for k, v in p.items()}
        return p
