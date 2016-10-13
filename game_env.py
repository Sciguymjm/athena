from gym import spaces
from gym import Env
from rl.core import Space
import numpy as np
from numpy import random as ran
import random
import os
import memory_watcher
import menu_manager
import pad as p
import state as st
import state_manager
import stats as stat
import copy
import itertools
from pad import Button

def find_dolphin_dir():
    """Attempts to find the dolphin user directory. None on failure."""
    candidates = ['~/.dolphin-emu', '~/.local/share/.dolphin-emu']
    for candidate in candidates:
        path = os.path.expanduser(candidate)
        if os.path.isdir(path):
            return path
    return None

def write_locations(dolphin_dir, locations):
    """Writes out the locations list to the appropriate place under dolphin_dir."""
    path = dolphin_dir + '/MemoryWatcher/Locations.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(locations))

        dolphin_dir = find_dolphin_dir()
        if dolphin_dir is None:
            print('Could not detect dolphin directory.')
            return

def get_analog_from_index(num):
    if num == 0:
        analog_y = 1.0
        analog_x = 0.5
    elif num == 1:
        analog_y = 0
        analog_x = 0.5
    elif num == 2:
        analog_y = 0.5
        analog_x = 0
    elif num == 3:
        analog_y = 0.5
        analog_x = 1.0
    elif num == 4: # no movement
        analog_x = 0.5
        analog_y = 0.5
    return analog_x,  analog_y



class ButtonSpace(Space):
    buttons = [Button.A,  Button.B,  Button.X,  Button.Y]
    def contains(self,  x):
        return x in self.buttons

    def sample(self, seed=None):
        return random.choice(self.buttons)


class AnalogSpace(Space):
    bounds = [0.0,  1.0]

    def contains(self,  x):
        return self.bounds[0] <= x <= self.bounds[1]

    def sample(self,  x):
        return random.randrange(0,  100) / 100.0

class MeleeEnv(Env):
    reward_range = (-1,  1)
    action_space = spaces.MultiDiscrete([[0, 4], [0, 4],  [0, 1], [0, 1], [0, 1], [0, 1], [0,  10],  [0,  10]])
    # main stick, c stick, a, b, x, y, l %, r %
    observation_space = np.array([0] * 22)
    

    last_action = 0
    def __init__(self):
        self.action_list = []
        self.mm = None
        self.mw = None
        self.state = None
        self.stats = None
        self.sm = None
        self.pad = None
        self.previous = None
        self.last_frame = None
        self.buttons = [Button.A,  Button.B,  Button.X,  Button.Z,  Button.NA]
        self.buttonStates = {Button.A: False,  Button.B: False,  Button.X: False,  Button.Z: False,  Button.NA: False}
        sticks = np.linspace(0, 1, 3)
        stickCombos = itertools.product(sticks,  repeat=2)

        self.action_space = np.array([[x,  y[0],  y[1]] for y in stickCombos for x in self.buttons ])
        print (self.action_space)
    def send_controller(self,  action):
            button,  analog_x,  analog_y = self.action_space[np.argmax(action)]
            print (button,  analog_x,  analog_y)
            if not self.buttonStates[button]:
                if button == Button.NA or button is None:
                    for b in self.buttons:
                        if b != Button.NA:
                            print ("Resetting all because N/A")
                            self.action_list.append((0,  self.pad.release_button,  [b]))
                            self.buttonStates[b] = False
                    #self.action_list.append((3,  None,  []))
                else:
                    print ("Appending",  button)
                    self.action_list.append((0,  self.pad.press_button,  [button]))
                    self.buttonStates[button] = True
            for b in self.buttons:
                if button != b and b != Button.NA:
                    self.action_list.append((0,  self.pad.release_button,  [b]))
                    self.buttonStates[b] = False
            self.action_list.append((0,  self.pad.tilt_stick,  [p.Stick.MAIN,  analog_x,  analog_y]))
            

            self.action_list.append((3,  None, []))
        
    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
#        self.action_list.append((0,  pad.tilt_stick,  [p.Stick.MAIN,  analog_x,  analog_y]))
#            if button == 3 or button is None:
#                self.action_list.append((3,  None,  []))
#            else:
#                self.action_list.append((0,  pad.press_button,  [Button(button)]))
#                self.action_list.append((3,  pad.release_button,  [Button(button)]))
#            
#            self.action_list.append((1,  None, []))
        while self.state.menu != st.Menu.Game:
            self.last_frame = self.state.frame
            while self.state.frame <= self.last_frame:
                #self.last_frame = self.state.frame
                res = next(self.mw)
                if res is not None:
                    self.sm.handle(*res)
        if self.state.menu == st.Menu.Game:
            #advance
            #fox.advance(state, pad,  model)'
        
            self.send_controller(action)
            self.previous = copy.deepcopy(self.state)
            print("done with frame")
            while self.action_list:
                wait, func, args = self.action_list[0]
                if self.state.frame - self.last_action < wait:
                    self.last_frame = self.state.frame
                    while self.state.frame < self.last_frame + wait:
                        res = next(self.mw)
                        if res is not None:
                            self.sm.handle(*res)
                           # print (res)
                else:
                    self.action_list.pop(0)
                    if func is not None:
                        func(*args)
                    self.last_action = self.state.frame
#        else:
#            print("stall")
#            return self.stall(action)
        reward = self.get_reward(self.state,  self.previous)

        
        print ("action",  action)
        #print("Final",  self.previous,  self.state,  self.get_data(self.state), reward)
        return self.get_data(copy.deepcopy(self.state)), reward, False, {}

    def stall(self,  action):
        while self.state.menu != st.Menu.Game:
            self.last_frame = self.state.frame
            while self.state.frame <= self.last_frame:
                #self.last_frame = self.state.frame
                res = next(self.mw)
                if res is not None:
                    self.sm.handle(*res)
            if self.state.menu == st.Menu.Characters:
                self.mm.pick_fox(self.state, self.pad)
            elif self.state.menu == st.Menu.Stages:
                # Handle this once we know where the cursor position is in memory.
                self.pad.tilt_stick(p.Stick.C, 0.5, 0.5)
            elif self.state.menu == st.Menu.PostGame:
                self.mm.press_start_lots(self.state, self.pad)

        return self.step(action)

    def reset(self):
        try:
            os.system("killall -s KILL dolphin-emu")
        except:
            pass
        dolphin_dir = find_dolphin_dir()
        if dolphin_dir is None:
            print('Could not find dolphin config dir.')
            return

        self.state = st.State()
        self.sm = state_manager.StateManager(self.state)
        write_locations(dolphin_dir, self.sm.locations())

        self.stats = stat.Stats()

        print('Start dolphin now. Press ^C to stop ')
        from subprocess import Popen
        Popen(["dolphin-emu", "--movie=/home/sci/workspace/Athena/falcon.dtm",  "--exec=/home/sci/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"])
        self.mw = memory_watcher.MemoryWatcher(dolphin_dir + '/MemoryWatcher/MemoryWatcher')
        self.pad = p.Pad(dolphin_dir + '/Pipes/p3')
        #run(state, sm, mw, pad, stats)
        self.mm = menu_manager.MenuManager()
        self.last_frame = self.state.frame
        print(vars(self))
#        while self.state.frame <= self.last_frame:
#            self.last_frame = self.state.frame
#            res = next(self.mw)
#            if res is not None:
#                self.sm.handle(*res)
#               # print (res)
#        print(vars(self))
        return self.get_data(self.state)


    def get_data(self,  data):
        enemy = data.players[0]
        me = data.players[2]
        d = []
        for them in [enemy, me]:
            [d.append(v) for v in [them.jumps_used,  them.attack_vel_x,  them.attack_vel_y,  them.self_air_vel_x,  them.self_air_vel_y,  them.shield_size,  them.facing,  them.action_state,  them.percent,  them.pos_x,  them.pos_y]]
        return np.array([d])

    def get_reward(self,  state,  prev):
        if prev is None:
            return 0.0
        reward = 0.0
        them = state.players[0] # 1st
        me = state.players[2] # 3rd

        them_p = prev.players[0]
        me_p = prev.players[2]
        #print("Percent",  float(them.percent) - them_p.percent, them.percent,  me.percent,  them_p.percent)
        diff = float(them.percent) - them_p.percent
        #if diff != 0:
            #print ("diff",  diff)
        if int(them.action_state) <= int(0x000A) and int(them.action_state) != int(them_p.action_state): # dead
            reward += 1
        elif diff > 0:
            reward += diff * 0.05
        

        if int(me.action_state) <= int(0x000A) and int(me.action_state) != int(me_p.action_state): # dead
            reward -= 1
        elif (me.percent - me_p.percent > 0):
            reward -= (me.percent - me_p.percent) * 0.05
        
        return reward

    def render(self,  mode=None,  close=None):
        pass
