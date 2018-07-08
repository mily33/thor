import h5py
import numpy as np


class Env(object):
    def __init__(self, config = dict()):
        self.action_list = ['MoveAhead', 'MoveBack', 'RotateLeft', 'RotateRight']
        self.scene_name = config.get('scene_name', 'bedroom_04')
        self.file_path = config.get('file_path', "data/%s.h5" % self.scene_name)
        self.terminal_id = config.get('terminal_id', 0)
        self.file = h5py.File(self.file_path, 'r')

        self.locations = self.file['location'][()]
        self.rotations = self.file['rotation'][()]
        self.n_location = self.locations.shape[0]

        self.terminal = np.zeros(self.n_location)
        self.terminal[self.terminal_id] = 1
        self.terminal_state, = np.where(self.terminal)

        self.transition_graph = self.file['graph'][()]
        self.shortest_path_distance = self.file['shortest_path_distance'][()]

        self.s_t = np.zeros([2048])

    def step(self, action):
