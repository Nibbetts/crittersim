import numpy as np
from critters.critter_base import Critter

class CritterRandom(Critter, object):

    TURN_SECONDS = 2
    ACCEL_SECONDS = 1

    TURN_CHANCE = 1.0 / (TURN_SECONDS*60) # Assume 60 FPS
    ACCEL_CHANCE = 1.0 / (ACCEL_SECONDS*60)
    MAX_ROT = np.pi / Critter.MAX_ROT_FRAC

    score1000 = 0
    scores = []
    average = 0

    @staticmethod
    def class_step():
        #if Critter.steps % 1000 == 0:
            #CritterRandom.scores.append(CritterRandom.score1000)
            #CritterRandom.average = np.mean(CritterRandom.scores)
            #print(CritterRandom.score1000)
            #CritterRandom.score1000 = 0
        pass

    def __init__(self, x, y, image, world, vel_x=None, vel_y=None, size=Critter.BIRTH_SIZE):
        super(CritterRandom, self).__init__(x, y, image, world, vel_x, vel_y, size)

        self.target_angle = (2*np.random.random() - 1) * np.pi
        self.accel_toggle = False

    # The following are functions which change according to AI type:
    def _act(self):
        if np.random.random() < CritterRandom.TURN_CHANCE:
            self.target_angle = (2*np.random.random() - 1) * np.pi
        angle_diff = self.least_angle_to(self.angle, self.target_angle)
        self.rotate = np.sign(angle_diff)
        if abs(angle_diff) <= CritterRandom.MAX_ROT:
            self.rotate *= abs(angle_diff)/CritterRandom.MAX_ROT
            if np.random.random() < CritterRandom.ACCEL_CHANCE: self.accel_toggle = True
        elif np.random.random() < CritterRandom.ACCEL_CHANCE: self.accel_toggle = False
        if np.random.random() < CritterRandom.ACCEL_CHANCE:
            self.accel_toggle = not self.accel_toggle

        return self.rotate, self.accel_toggle

    def _train(self, reward):
        CritterRandom.score1000 += reward

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
