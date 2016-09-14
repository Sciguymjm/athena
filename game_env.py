import gym
from gy import spaces
from gym.utils import seeding
import numpy as np
from os import path


class MeleeEnv(gym.Env):
    metadata = {
        'video.frames_per_second': 60
    }
    
    def __init__(self):
        self.observation_space = spaces.Discrete(8)
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(5)))
        self._seed()
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
