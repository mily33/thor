import tensorflow as tf
import tensorflow.contrib.slim as slim


class ActorCritic(object):
    def __init__(self, scope, action_size):
        self.scope = scope
        self.action_size = action_size

    def prepare_loss(self, ):
        # TODO