import pad as p
from pad import Button
import numpy as np
import random
import copy
from p3 import memory
buttons = [Button.A,  Button.B,  Button.X,  None]
def get_reward(state,  prev):
    reward = 0.0
    them = state.players[0] # 1st
    me = state.players[2] # 3rd
    
    them_p = prev.players[0]
    me_p = prev.players[2]
    #print("Percent",  float(them.percent) - them_p.percent, them.percent,  me.percent,  them_p.percent)
    diff = float(them.percent) - them_p.percent
    #if diff != 0:
        #print ("diff",  diff)
    reward += diff * 0.05
    if int(them.action_state) <= int(0x000A) and int(them.action_state) != int(them_p.action_state): # dead
        reward += 1
    
    
    reward -= (me.percent - me_p.percent) * 0.01
    if int(me.action_state) <= int(0x000A) and int(me.action_state) != int(me_p.action_state): # dead
        reward -= 1
    return reward

def get_data(prev):
    d = np.array([])
    for x in prev:
        them = x.players[0]
        me = x.players[2]
        
        values = [them.action_state,  them.percent,  them.pos_x,  them.pos_y,  me.action_state,  me.percent,  me.pos_x,  me.pos_y]
        #values = list(vars(x.players[0]).values()) +  list(vars(x.players[2]).values())
        d = np.append(d,  np.array(values).reshape((1,  -1)))
    return d

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

class Fox:
    def __init__(self):
        self.action_list = []
        self.last_action = 0
        self.prev = []
        self.previous_button = [0,  0, 0 , 0]
        self.previous_analog = [0.5,  0.5]
    def advance(self, state, pad,  model):
        while self.action_list:
            wait, func, args = self.action_list[0]
            if state.frame - self.last_action < wait:
                return
            else:
                self.action_list.pop(0)
                if func is not None:
                    func(*args)
                self.last_action = state.frame
        else:
#        else:
#            # Eventually this will point at some decision-making thing.
#            self.wavedash(pad)
        #print(vars(state.players[0]).values())
            #reward = 0
            analog_x,  analog_y = (0.5,  0.5)
            num = 0
            #data = list(vars(state.players[0]).values()) +  list(vars(state.players[2]).values())
            if len(self.prev) == memory:
                reward = get_reward(state, self.prev[-1])
                #print("Reward:",  reward)

                #print(self.previous_button,  np.argmax(self.previous_button),  reward)
                self.previous_button[np.argmax(self.previous_button)] += reward
                self.previous_analog[np.argmax(self.previous_analog)] += reward
                self.previous_button = np.append(self.previous_button,  self.previous_analog)
                self.prev.pop(0)
                self.prev.append(copy.deepcopy(state))
                #print (self.prev[0] == self.prev[1])
                if reward != 0.0:
                    d = np.array([])
                    d = get_data(self.prev)
                    model.fit(np.array(d).reshape((1,  -1)), self.previous_button.reshape((1,  -1)), nb_epoch=1)
            elif len(self.prev) < memory:
                self.prev.append(copy.deepcopy(state))
                return
            if random.random() < 0.05:
                button = random.choice(buttons)
                self.previous_button = np.array([(x == button) for x in buttons])
                #analog_x = random.randrange(0,  10)/10.0
                #analog_y = random.randrange(0,  10)/10.0
                analog_x,  analog_y = get_analog_from_index(random.randrange(0,  4))
            else:
                #print(np.array(data).shape)
                d = get_data(self.prev)
                #data = np.array(data).reshape((1,  -1))
                #print(data.shape)
                a = model.predict(d.reshape((1,  -1)))
                #print (a[0])
                a1 = a[0][:4]
                
                button = np.argmax(a1)
                self.previous_button = np.array(a1)
                analog_x, analog_y = (0.5,  0.5)
                stick = np.argmax(a[0][4:])
                self.previous_analog = a[0][4:]
                analog_x,  analog_y = get_analog_from_index(stick)
                #[analog_x,  analog_y][np.argmax(a[4:6])] = 1.0
                #analog_x = float(a[0][4])
                #analog_y = float(a[0][5])
            
            self.action_list.append((0,  pad.tilt_stick,  [p.Stick.MAIN,  analog_x,  analog_y]))
            if button == 3 or button is None:
                self.action_list.append((3,  None,  []))
            else:
                self.action_list.append((0,  pad.press_button,  [Button(button)]))
                self.action_list.append((3,  pad.release_button,  [Button(button)]))
            
            self.action_list.append((1,  None, []))
            #print ("Stick",  num,  analog_x,  analog_y)
            #self.prev = state
            #self.previous_analog = [analog_x,  analog_y]

    def shinespam(self, pad):
        self.action_list.append((0, pad.tilt_stick, [p.Stick.MAIN, 0.5, 0.0]))
        self.action_list.append((0, pad.press_button, [p.Button.B]))
        self.action_list.append((1, pad.release_button, [p.Button.B]))
        self.action_list.append((0, pad.tilt_stick, [p.Stick.MAIN, 0.5, 0.5]))
        self.action_list.append((0, pad.press_button, [p.Button.X]))
        self.action_list.append((1, pad.release_button, [p.Button.X]))
        self.action_list.append((1, None, []))
    def wavedash(self,  pad):
        actions = [(0,  pad.press_button, [p.Button.X]), 
        (1,  pad.tilt_stick, [p.Stick.MAIN,  0,  0.45]), 
        (0,  pad.release_button, [p.Button.X]), 
        (10,  pad.press_trigger,  [p.Trigger.L,  1]), 
        (15, pad.press_trigger,  [p.Trigger.L,  0]), 
        (0,  pad.tilt_stick,  [p.Stick.MAIN,  0.5, 0.5]), 
        (25,  None,  [])]
        for action in actions:
            self.action_list.append(action)
        
