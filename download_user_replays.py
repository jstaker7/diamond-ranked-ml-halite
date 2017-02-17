"""
WARNING: This script has not been cleaned up and is provided in the raw for
historical purposes only. I believe the replays are no longer available on the
halite server anyway, and are provided in the repository.
"""
from datetime import datetime
import shutil
import requests
import json
import numpy as np
import warnings
from time import sleep
import os

import gzip, zlib

user = "2697" # Nmalaguti
FOLDER = 'ng_replays_54' # 47-54
PNAME = 'nmalaguti'


print(FOLDER)

if not os.path.exists('/Users/Peace/Projects/halite/replays/'+FOLDER):
    os.mkdir('/Users/Peace/Projects/halite/replays/'+FOLDER)
else:
    raise FolderAlreadyExists

def get_replay(tup):

    filename = tup[1]
    sleep(0.1)
    url = 'http://s3.amazonaws.com/halitereplaybucket/{0}'.format(filename)

    response = requests.get(url, stream=True)

    try:
        raw = zlib.decompress(response.raw.data, 15+32)
    except:
        try:
            raw = gzip.decompress(response.raw.data)
        except:
            print('Replay skipped: {0}'.format(filename))
            return None, None, None
    obj = json.loads(raw.decode('utf-8'))

    return obj


def parse_replay(replay_json):
    try:
        frames = np.array(replay_json['frames'])
    except Exception as e:
        print(e)
        return False, None, None
    moves = np.array(replay_json['moves'])

    # We only want replays that are worthwhile
    if not frames.shape[0] > 15:
        return False, None, None

    for i in range(len(replay_json['player_names'])):
        if PNAME in replay_json['player_names'][i]:
            winner = i+1
            break

    # Rescale strengths so it is easier to retain ownership information
    frames[:, :, :, 1] -= (frames[:, :, :, 1] == 255) # Avoid overflow
    frames[:, :, :, 1] += 1 # Shift everything by one
    
    own_bool = (frames[:, :, :, 0] == winner)

    neu_array = (frames[:, :, :, 0] == 0) * frames[:, :, :, 1]
    own_array = own_bool * frames[:, :, :, 1]
    ene_array = (np.logical_not(neu_array + own_array)) * frames[:, :, :, 1]
    
    str_array = np.array([neu_array, own_array, ene_array])
    
    pro_array = np.array(replay_json['productions']) * 15 # Rescaled
    
    # Not very efficient, but works for now
    tiled = np.tile(pro_array, [frames.shape[0], 1, 1])
    production = np.expand_dims(tiled, axis=3)
    
    strengths = np.rollaxis(str_array, 0, 4)
    
    stacked = np.concatenate([strengths, production], axis=3)
 
    return True, stacked.astype('uint8'), moves * own_bool[:-1, :, :]

request = "https://halite.io/api/web/game?userID={}&limit=1000&versionNumber={}".format(user, FOLDER.split('_')[-1])
r = requests.get(request)
game_ids = [game['replayName'] for game in r.json()]


for ix, game_id in enumerate(game_ids):
    json_data = get_replay(('', game_id))
    
    if ix == 0:
        print(json_data['player_names'])

    quality, replay, moves = parse_replay(json_data)

    if quality:
        outfilename = '/Users/Peace/Projects/halite/replays/'+FOLDER+'/{0}.npz'

        np.savez_compressed(outfilename.format(str(ix)), replay=replay, moves=moves)





