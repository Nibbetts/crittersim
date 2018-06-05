import numpy as np
from critters.critter_base import Critter

class CritterPlayer(Critter, object):

    def __init__(self, x, y, image, world,
                 vel_x=None, vel_y=None, size=Critter.BIRTH_SIZE, key_list=None):
        super(CritterPlayer, self).__init__(x, y, image, world, vel_x, vel_y, size)
        if key_list is None: self.key_list = [False, False, False, False]
        else: self.key_list = key_list
        # NOTE: key_list = [left, right, up, down], all booleans.

    # The following are functions which change according to AI type:
    def _act(self):
        #if key_list[0] == key_list[1]: rotate = 0
        #else:
        #    rotate = -1 if key_list[0] else 1
        #return rotate, 0
        return self.key_list[0]-self.key_list[1], self.key_list[2]-self.key_list[3]

    def _train(self, reward):
        #loss = -reward
        pass

    # Overriding parent here:
    def reproduce(self):
        self.__class__(self.x + np.random.random()*Critter.IMAGE_SIZE - Critter.HALF_SIZE,
                self.y + np.random.random()*Critter.IMAGE_SIZE - Critter.HALF_SIZE,
                self.original_image, self.world,
                self.vel_x + np.random.random()*2 - 1,
                self.vel_y + np.random.random()*2 - 1,
                key_list=self.key_list, size=0.4)
        self.kill(True)

    def kill(self, reproduced=False):
        super(CritterPlayer, self).kill(reproduced)
        if not reproduced:
            self.__class__(np.random.randint(self.world.world_width),
                np.random.randint(self.world.world_height),
                self.original_image, self.world,
                key_list=self.key_list, size=0.4)

    # # The following are empty for this class:
    #
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
