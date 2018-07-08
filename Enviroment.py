import h5py
import numpy as np
from constants import *
import random


class Env(object):
    def __init__(self, config=dict()):
        self.action_list = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight']
        self.scene_name = config.get('scene_name', 'bedroom_04')
        self.file_path = config.get('file_path', "data/%s.h5" % self.scene_name)
        self.terminal_id = config.get('terminal_id', 0)
        self.init_state_id = config.get('initial_state', None)
        self.file = h5py.File(self.file_path, 'r')

        self.locations = self.file['location'][()]
        self.rotations = self.file['rotation'][()]
        self.n_location = self.locations.shape[0]
        self.feature = self.file['resnet_feature'][()]

        self.terminal = np.zeros(self.n_location)
        self.terminal[self.terminal_id] = 1
        self.terminal_state, = np.where(self.terminal)

        self.transition_graph = self.file['graph'][()]
        self.shortest_path_distance = self.file['shortest_path_distance'][()]

        self.history_len = HISTORY_LENGTH
        self.s_t = np.zeros([2048, self.history_len])
        self.s_t1 = np.zeros_like(self.s_t)
        self.target = self._tile_state(self.terminal_id)

        self.reward = 0
        self.terminated = False
        self.collided = False
        self.current_state_id = None

        if self.init_state_id is not None:
            assert self.terminal[self.init_state_id] == 0, 'Initial state is exactly the terminal state!'
            assert self.shortest_path_distance[self.init_state_id][self.terminal_id] >= 0, \
                'Target state is unreachable to the initial state!'
            self.current_state_id = self.init_state_id
            self.s_t = self._tile_state(self.current_state_id)
        else:
            self.reset()

    def reset(self):
        while True:
            k = random.randrange(self.n_location)
            if self.terminal[k] != 1 and self.shortest_path_distance[k][self.terminal_id] != -1:
                self.current_state_id = k
                break
        self.s_t = self._tile_state(self.current_state_id)

    def _tile_state(self, observation):
        k = random.randrange(10)
        return self.feature[observation][k][:, np.newaxis]

    def step(self, action):
        if self.transition_graph[self.current_state_id][action] != -1:
            self.current_state_id = self.transition_graph[self.current_state_id][action]
            if self.terminal[self.current_state_id] == 1:
                self.terminated = True
                self.collided = False
            else:
                self.terminated = False
                self.collided = False
        else:
            self.collided = True
            self.terminated = False

        self.reward = self._reward(self.terminated, self.collided)
        self.s_t = self._tile_state(self.current_state_id)
        self.s_t1 = np.append(self.s_t[:, 1:], self.s_t, axis=1)

    @staticmethod
    def _reward(terminal, collided):
        if terminal:
            return 10.0
        elif collided:
            return -0.1
        else:
            return -0.01



