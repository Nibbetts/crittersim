#from interface import World
import numpy as np
import pygame
#from pygame.locals import *
import time
import critters as zoo

class Game(World):

    def __init__(self, window_width=1024,
                 window_height=1024, 
                 #wrap = False,
                 num_per_species=3):
        self.window_width = window_width
        self.window_height = window_height
        #self.wrap = wrap
        self.num_per_species = num_per_species

    def clear_state(self):
        return

    def draw_object(self, obj):
        return

    def start(self):
        self.time = 0

	    i = 0
        while i < self.enemy_count:
            enemy_candidate = pygame.Rect(np.random.randint(0,self.game_width-s$
            if not enemy_candidate.colliderect(self.agent):
                i += 1
                self.enemy.append(enemy_candidate)


        #Initialize Everything
        pygame.init()
        self.screen = pygame.display.set_mode((self.game_width, self.game_heigh$

        self.background = pygame.Surface((self.game_width, self.game_height))
        self.background = self.background.convert()
        self.background.fill(170)
        self.draw()
        return

    def draw(self):

        self.background.fill(160)
        pygame.draw.rect(self.background, 220, self.agent, 0)
        for e in self.enemy:
            pygame.draw.rect(self.background, 0, e, 0)
        pygame.draw.rect(self.background, 255, self.target, 0)
        return

    def is_running(self):
        return self.is_playing

    def get_state(self):
        arr = pygame.surfarray.pixels2d(self.background)
        return arr

    def reset(self):
        self.start()
        return

    def load(self):
        return