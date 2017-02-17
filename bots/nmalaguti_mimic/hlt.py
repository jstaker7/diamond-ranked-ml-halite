"""
Modified erdman starter
"""

import sys
from collections import namedtuple
from itertools import chain
import numpy as np

Square = namedtuple('Square', 'x y owner strength production')

Move = namedtuple('Move', 'square direction')

class GameMap:
    def __init__(self, size_string, production_string, map_string=None, myid=None):
        self.width, self.height = tuple(map(int, size_string.split()))
        self.production = np.array([int(pr) for pr in production_string.split()]) * 15/255. - 0.5
        self.contents = None
        self.myid = myid
        self.initial_map = self.get_frame(map_string)
    
    def get_frame(self, map_string=None):
        """
        Updates the map information from the latest frame provided by the
        Halite game environment. Custom-built to organize the data as
        efficiently as possible into the format expected by the model.
        """
        if map_string is None:
            map_string = get_string()
        split_string = map_string.split()
        
        owners = list()
        while len(owners) < self.width * self.height:
            counter = int(split_string.pop(0))
            owner = int(split_string.pop(0))
            owners.extend([owner] * counter)

        owners = np.array(owners)
        strengths = np.array([int(st) for st in split_string])
        strengths -= (strengths == 255)
        strengths += 1
        
        neu_array = (np.array(owners) == 0)
        own_array = (np.array(owners) == self.myid)
        ene_array = np.logical_not(neu_array + own_array)
        
        owner_array = own_array.astype('float32') - ene_array.astype('float32')
        
        frame = np.stack([strengths/255. - 0.5, owner_array, self.production], axis=-1)
        
        return frame.reshape((self.height, self.width, 3))


    def __iter__(self):
        "Allows direct iteration over all squares in the GameMap instance."
        return chain.from_iterable(self.contents)


###############################################################################
# Functions for communicating with the Halite game environment (formerly
# contained in separate module networking.py)
###############################################################################


def send_string(s):
    sys.stdout.write(s)
    sys.stdout.write('\n')
    sys.stdout.flush()


def get_string():
    return sys.stdin.readline().rstrip('\n')


def get_init():
    playerID = int(get_string())
    m = GameMap(get_string(), get_string(), myid = playerID)
    return playerID, m


def send_init(name):
    send_string(name)


def send_frame2(moves):
    send_string(' '.join(str(move.square.x) + ' ' + str(move.square.y) + ' ' + str(move.direction) for move in moves))
