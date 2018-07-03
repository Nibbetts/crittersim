import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from critters.critter_base import Critter

class CritterFF(Critter, nn.Module, object):

    #W = load_weights()
    IN_FEATURES = 0
    OUT_FEATURES = 0

    def __init__(self, x, y, image, world,
                 vel_x=None, vel_y=None, size=Critter.BIRTH_SIZE):
        super(CritterFF, self).__init__(x, y, image, world, vel_x, vel_y, size)
        super(CritterFF, self).__init__()
        self.fc1 = nn.Linear(1 * 2 * 3, 50)
        self.fc2 = nn.Linear(120, 84)

    # The following are functions which change according to AI type:
    def _act(self):
        return 0, 0
        #return rotate, accelerate

    def _train(self, reward):
        #loss = -reward
        pass

    # @staticmethod
    # def class_step():
    #     pass
    #
    # @staticmethod
    # def reset_weights():
    #     pass
    # 
    # @staticmethod
    # def save_weights():
    #     pass
    #
    # @staticmethod
    # def load_weights():
    #     pass
