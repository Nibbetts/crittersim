import numpy as np
from critters.critter_base import Critter

class CritterFF(Critter, object):

    #W = load_weights()

    def __init__(self, x, y, image, world,
                 vel_x=None, vel_y=None, size=Critter.BIRTH_SIZE):
        super(CritterFF, self).__init__(x, y, image, world, vel_x, vel_y, size)

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
