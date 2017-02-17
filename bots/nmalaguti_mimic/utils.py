import os

import numpy as np
import tensorflow as tf

def prepare_filepaths(dataset_path):
    paths = []
    for root, dirs, files in os.walk(dataset_path):
        path = root.split('/')
        for file in files:
            if '.' != file[0]:
                if 'numpy' not in file:
                    paths.append(root+'/'+file)
    return paths

def pad_replay(replay, dims):
    old_shape = replay.shape
    
    old_pad1y = 0
    old_pad2y = 0
    
    old_pad1x = 0
    old_pad2x = 0
    
    pad_amounty = dims - replay.shape[1]
    side1_pady = pad_amounty//2
    side2_pady = pad_amounty - side1_pady

    old_pad1y += min(old_shape[1], side1_pady)
    old_pad2y += min(old_shape[1], side2_pady)
    
    pad_amountx = dims - replay.shape[2]
    side1_padx = pad_amountx//2
    side2_padx = pad_amountx - side1_padx

    old_pad1x += min(old_shape[2], side1_padx)
    old_pad2x += min(old_shape[2], side2_padx)

                            
    # Padding by reflection
    frame = np.concatenate([replay[:, -old_pad1y:, :, :], replay], axis=1)
    frame = np.concatenate([frame, replay[:, :old_pad2y, :, :]], axis=1)
    replay = frame.copy()
    frame = np.concatenate([replay[:, :, -old_pad1x:, :], frame], axis=2)
    frame = np.concatenate([frame, replay[:, :, :old_pad2x, :]], axis=2)
    

    
    pad_amounty = dims - frame.shape[1]
    side1_pady = pad_amounty//2
    side2_pady = pad_amounty - side1_pady
    old_pad1y += side1_pady
    old_pad2y += side2_pady
    
    pad_amountx = dims - frame.shape[2]
    side1_padx = pad_amountx//2
    side2_padx = pad_amountx - side1_padx
    old_pad1x += side1_padx
    old_pad2x += side2_padx

    frame = np.pad(frame, ((0,0), (side1_pady, side2_pady),
                                (side1_padx, side2_padx), (0,0)),
                                mode='constant', constant_values=0)

    return frame, ((old_pad1y, old_pad2y), (old_pad1x, old_pad2x))

def aug(move, frame, iter, frames_padding):
    """
    Augment the data using rotations and mirroring.
    """
    should_mirror = iter>3
    
    frames_padding = np.array([[frames_padding[0][0], frames_padding[0][1]],
                                [frames_padding[1][0], frames_padding[1][1]]])
    

    num_rotate = iter%4
    
    if should_mirror:
        frame = np.flip(frame, 0)
        move = np.flip(move, 0)
    
        frames_padding = np.array([[frames_padding[0][1], frames_padding[0][0]],
                                    [frames_padding[1][0], frames_padding[1][1]]])
    
    frame = np.rot90(frame, num_rotate)
    move = np.rot90(move, num_rotate)
    
    for _ in range(num_rotate):
        frames_padding = np.array([[frames_padding[1][1], frames_padding[1][0]],
                                   [frames_padding[0][0], frames_padding[0][1]]])
    
    # We need to change the move values to match the augmentation
    # STILL will not change; this is simply reordering the axis
    # Current axis: N:1, E:2, S:3, W:4
    if should_mirror:
        move = move[:, :, (0, 3, 2, 1, 4)]

    shift = np.roll(move[:, :, 1:].copy(), -1*num_rotate, 2)
    move[:, :, 1:] = shift

    return move, frame, frames_padding

def relu(x, alpha=0., max_value=None):
    '''ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def convolve(input, i_units, o_units, name, activate=True, kernel=3):
    conv1_weights = tf.get_variable(name, [kernel, kernel, i_units, o_units],
                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv1_bias = tf.get_variable("{0}_b".format(name),
                    initializer=tf.constant(0., shape=[o_units]))
    pre = tf.nn.conv2d(input, conv1_weights,
                            strides=[1, 1, 1, 1], padding='SAME') + conv1_bias
                            
    if activate:
        h_conv1 = relu(pre, 0.3, 50) # Very leaky ReLU with 0.3 slope
    else:
        h_conv1 = pre

    return h_conv1

def dont_overconfidence(frame, actions):
    
    still = actions == 0
    north = actions == 1
    east = actions == 2
    south = actions == 3
    west = actions == 4
    
    frame[:, :, 0] = frame[:, :, 0] + 0.5

    my_strengths = (frame[:, :, 1] > 0) * (frame[:, :, 0])
    
    total_strengths = my_strengths.astype('float32') * still
    total_strengths += np.roll(my_strengths * north, -1, axis=0)
    total_strengths += np.roll(my_strengths * east, 1, axis=1)
    total_strengths += np.roll(my_strengths * south, 1, axis=0)
    total_strengths += np.roll(my_strengths * west, -1, axis=1)
    
    neutrals = (frame[:, :, 1].astype('int8') == 0) * (frame[:, :, 0])
    
    enemeies1 = frame[:, :, 1] < 0
    enemeies = np.roll(enemeies1, -1, axis=0)
    enemeies += np.roll(enemeies1, 1, axis=1)
    enemeies += np.roll(enemeies1, 1, axis=0)
    enemeies += np.roll(enemeies1, -1, axis=1)
    
    neutral_wall = neutrals.astype('float32')*255 > 1.000001
    
    bad_moves = (neutrals.astype('float32') - total_strengths.astype('float32')) > -0.00000001
    
    bad_moves += total_strengths > 1.95
    
    bad_moves += (neutral_wall*enemeies) > 0
    
    bad_moves = bad_moves > 0
    
    dont_move_north = np.roll(bad_moves, 1, axis=0)
    dont_move_east = np.roll(bad_moves, -1, axis=1)
    dont_move_south = np.roll(bad_moves, -1, axis=0)
    dont_move_west = np.roll(bad_moves, 1, axis=1)
    
    n_adj = dont_move_north * north
    e_adj = dont_move_east * east
    s_adj = dont_move_south * south
    w_adj = dont_move_west * west
    
    north -= n_adj
    east -= e_adj
    south -= s_adj
    west -= w_adj
    
    new_actions = np.zeros(actions.shape) + north + 2*east + 3*south + 4*west

    return new_actions.astype('int8')

