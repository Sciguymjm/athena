import os.path
import memory_watcher
import menu_manager
import pad as p
import state as st
import state_manager
import stats as stat
import random 
import game_env
from keras.layers import merge,  Convolution2D,  Flatten,  Dense,  Input,  Activation
from keras.models import Sequential,  Model
from keras.optimizers import RMSprop
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
import sys, os

memory = 10
nb_actions = 9
def build_network(env,  nb_actions):
#    model = Sequential()
#    model.add(Dense(128, input_shape=(8*memory, )))
#    model.add(Activation('relu'))
#    model.add(Dense(128))
#    model.add(Activation('softmax'))
#    #model.add(Dropout(0.2))
#
#    model.add(Dense(9)) # button index, analog x analog y
#    #model.add(Activation('relu')) 
#    model.compile(loss='mse', optimizer=sgd(lr=.2))
    actor = Sequential()
    actor.add(Flatten(input_shape=(1, ) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(9))
    actor.add(Activation('linear'))
    print(actor.summary())
    return actor
    
    
def build_critic(env,  nb_actions):
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1, ) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())
    return critic,  action_input

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

def run():
    env = game_env.MeleeEnv()
    actor = build_network(env,  nb_actions)
    critic,  action_input = build_critic(env,  nb_actions)
    memory = SequentialMemory(limit=100000)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3,  size=nb_actions)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      delta_range=(-10., 10.))
    agent.compile([RMSprop(lr=.001), RMSprop(lr=.001)], metrics=['mae'])
    
    agent.fit(env, nb_steps=1000000, visualize=True, verbose=1, nb_max_episode_steps=200000)
    # After training is done, we save the final weights.
    agent.save_weights('ddpg_{}_weights.h5f'.format(str(random.randrange(0, 100000))), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    #agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
    
    
    
    
#    print("fox")
#    mm = menu_manager.MenuManager()
#    while True:
#        last_frame = state.frame
#        res = next(mw)
#        if res is not None:
#            sm.handle(*res)
#           # print (res)
#        if state.frame > last_frame:
#            stats.add_frames(state.frame - last_frame)
#            start = time.time()
#            make_action(state, pad, mm, fox,  agent)
#            stats.add_thinking_time(time.time() - start)
#            #print ("action")

def make_action(state, pad, mm, fox,  model):
    if state.menu == st.Menu.Game:
        fox.advance(state, pad,  model)
    elif state.menu == st.Menu.Characters:
        mm.pick_fox(state, pad)
    elif state.menu == st.Menu.Stages:
        # Handle this once we know where the cursor position is in memory.
        pad.tilt_stick(p.Stick.C, 0.5, 0.5)
    elif state.menu == st.Menu.PostGame:
        mm.press_start_lots(state, pad)

def main():
    run()
#    dolphin_dir = find_dolphin_dir()
#    if dolphin_dir is None:
#        print('Could not find dolphin config dir.')
#        return
#
#    state = st.State()
#    sm = state_manager.StateManager(state)
#    write_locations(dolphin_dir, sm.locations())
#
#    stats = stat.Stats()
#
#    try:
#        print('Start dolphin now. Press ^C to stop ')
#        
#        Popen(["dolphin-emu", "--movie=/home/sci/workspace/Athena/falcon.dtm",  "--exec=/home/sci/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"])
#        mw = memory_watcher.MemoryWatcher(dolphin_dir + '/MemoryWatcher/MemoryWatcher')
#        pad = p.Pad(dolphin_dir + '/Pipes/p3')
#        run()
#    except KeyboardInterrupt:
#        print('Stopped')
#        print(stats)
from subprocess import Popen
if __name__ == '__main__':
    os.system("killall -s KILL dolphin-emu")
    main()
