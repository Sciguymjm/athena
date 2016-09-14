import os.path
import time

import fox as f
import memory_watcher
import menu_manager
import pad as p
import state as st
import state_manager
import stats as stat

from keras.layers import Convolution2D,  Flatten,  Dense,  Input,  Activation
from keras.models import Sequential
from keras.optimizers import sgd

memory = 10

def build_network(sm):
    model = Sequential()
    model.add(Dense(128, input_shape=(8*memory, )))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('softmax'))
    #model.add(Dropout(0.2))

    model.add(Dense(9)) # button index, analog x analog y
    #model.add(Activation('relu')) 
    model.compile(loss='mse', optimizer=sgd(lr=.2))
    return model
    
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

def run(state, sm, mw, pad, stats):
    model = build_network(sm)
    fox = f.Fox()
    print("fox")
    mm = menu_manager.MenuManager()
    while True:
        last_frame = state.frame
        res = next(mw)
        if res is not None:
            sm.handle(*res)
           # print (res)
        if state.frame > last_frame:
            stats.add_frames(state.frame - last_frame)
            start = time.time()
            make_action(state, pad, mm, fox,  model)
            stats.add_thinking_time(time.time() - start)
            #print ("action")

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
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print('Could not find dolphin config dir.')
        return

    state = st.State()
    sm = state_manager.StateManager(state)
    write_locations(dolphin_dir, sm.locations())

    stats = stat.Stats()

    try:
        print('Start dolphin now. Press ^C to stop ')
        
        Popen(["dolphin-emu", "--movie=/home/sci/workspace/Athena/falcon.dtm",  "--exec=/home/sci/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"])
        mw = memory_watcher.MemoryWatcher(dolphin_dir + '/MemoryWatcher/MemoryWatcher')
        pad = p.Pad(dolphin_dir + '/Pipes/p3')
        run(state, sm, mw, pad, stats)
    except KeyboardInterrupt:
        print('Stopped')
        print(stats)
from subprocess import Popen
if __name__ == '__main__':
    os.system("killall -s KILL dolphin-emu")
    main()
