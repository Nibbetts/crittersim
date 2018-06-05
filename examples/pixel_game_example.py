from interface import World
import numpy as np
import pygame
from pygame.locals import *
import time
import cv2

import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"

class Game(World):

    def __init__(self, game_width = 160,
                 game_height = 210, 
                 wrap = False,
                 enemy_count = 1, enemy_delay = 10,
                 enemy_size = 2, continuous_scoring = False,
                 zone_scoring = False, euclidean_scoring = True, 
                 time_penalty = True, vis=True):
        self.game_width = game_width
        self.game_height = game_height
        self.wrap = wrap
        self.enemy_count = enemy_count
        self.enemy_delay = enemy_delay
        self.enemy_size = enemy_size
        self.continuous_scoring = continuous_scoring
        self.zone_scoring = zone_scoring
        self.euclidean_scoring = euclidean_scoring
        self.time_penalty = time_penalty
        self.vis = vis

    def clear_state(self):
        return

    def draw_object(self, obj):
        return

    def start(self):
        self.score = 0
        self.time = 0
        self.is_playing = True
        self.zone_one_entered = False
        self.zone_two_entered = False
        self.zone_three_entered = False
        self.target = (np.random.randint(0,self.game_width-3),np.random.randint$
        self.agent = pygame.Rect(np.random.randint(0,self.game_width-3),np.rand$
        self.old_agent = (self.agent[0], self.agent[1])
        
        self.enemy = []

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
    def act(self, action):#return change in score.
        self.old_agent = (self.agent[0], self.agent[1])
        self.time += 1
        if action == 0:
            self.agent = pygame.Rect(self.agent[0] + 1, self.agent[1], self.age$
            if self.agent[0]+self.agent[3]-1 >= self.game_width:
                if self.wrap:
                    self.agent = pygame.Rect(0, self.agent[1], self.agent[2], s$
                else:
                    self.agent = pygame.Rect(self.agent[0] - 1, self.agent[1], $

        elif action == 1:
            self.agent = pygame.Rect(self.agent[0] - 1, self.agent[1], self.age$
            if self.agent[0] < 0:
                if self.wrap:
                    self.agent = pygame.Rect(self.game_width - self.agent[2], s$
                else:
                    self.agent = pygame.Rect(self.agent[0] + 1, self.agent[1], $
    
        elif action == 2:
            self.agent = pygame.Rect(self.agent[0], self.agent[1] + 1, self.age$
            if self.agent[1]+self.agent[2]-1 >= self.game_height:
                if self.wrap:
                    self.agent = pygame.Rect(self.agent[0], 0, self.agent[2], s$
                else:
                    self.agent = pygame.Rect(self.agent[0], self.agent[1] - 1, $

        elif action == 3:
            self.agent = pygame.Rect(self.agent[0], self.agent[1] - 1, self.age$
            if self.agent[1] < 0:
                if self.wrap:
                    self.agent = pygame.Rect(self.agent[0], self.game_height - $
                else:
                    self.agent = pygame.Rect(self.agent[0], self.agent[1] + 1, $
    
        if self.time % self.enemy_delay == 0:
            self.move_enemies()
    
        self.draw()
    
        if self.agent.collidelist(self.enemy) != -1:
            self.score -= 50
            self.is_playing = False
            return -50

    
        result = self.delta_score()
    
        if (self.agent[0] + self.agent[2] > self.target[0] and self.target[0] +$
            self.score += 500
            self.is_playing = False
            return 500

        return result


    def move_enemies(self):
        for i,e in enumerate(self.enemy):
            if self.agent[0] - e[0] >= self.enemy_size:
                e = pygame.Rect(e[0] + 1, e[1], e[2], e[3])
            elif self.agent[0] - e[0] < 0:
                e = pygame.Rect(e[0] - 1, e[1], e[2], e[3])
            if self.agent[1] - e[1] >= self.enemy_size:
                e = pygame.Rect(e[0], e[1] + 1, e[2], e[3])
            elif self.agent[1] - e[1] < 0:
                e = pygame.Rect(e[0], e[1] - 1, e[2], e[3])
            self.enemy[i] = e



    def delta_score(self):
        result = 0
        if self.continuous_scoring:
            result = -1
            if(abs(self.target[0] - self.old_agent[0]) > abs(self.target[0] - s$
                result = 1
            if(abs(self.target[1] - self.old_agent[1]) > abs(self.target[1] - s$
                result = 1
        elif self.zone_scoring:
            if abs(self.target[0] - self.agent[0]) < 10 and abs(self.target[1] $
                self.zone_one_entered = True
                result += 100
            if abs(self.target[0] - self.agent[0]) < 50 and abs(self.target[1] $
                self.zone_two_entered = True
                result += 50
            if abs(self.target[0] - self.agent[0]) < 90 and abs(self.target[1] $
                self.zone_three_entered = True
                result += 30
        elif self.euclidean_scoring:
            old_distance = np.sqrt((self.target[0] - self.old_agent[0])**2 + (s$
            new_distance = np.sqrt((self.target[0] - self.agent[0])**2 + (self.$
            result = old_distance - new_distance;

        if self.time_penalty:
            result -= 0.1
    
        self.score += result
        return result

    def get_time(self):
        return self.time

    def get_score(self):#score over the whole game
        return self.score

    def get_action_space(self):
        return range(4)

    def get_state_space(self):
        width, height, color = pygame.surfarray.array3d(self.background).shape
        return (width, height)

    def reset(self):
        self.start()
        return

    def load(self):
        return


