import tensorflow as tf
import numpy as np
import multiprocessing
import random
import os
from random import shuffle

from utils import pad_replay, prepare_filepaths, relu, convolve, aug

current_path = os.path.abspath(__file__)
replay_path = os.path.join('/', *current_path.split('/')[:-3], 'replays')

selected_replays = ['ng_replays_47',
                    'ng_replays_48',
                    'ng_replays_49',
                    'ng_replays_50',
                    'ng_replays_51',
                    'ng_replays_52',
                    'ng_replays_53',
                    'ng_replays_54']

_paths = [os.path.join(replay_path, p) for p in selected_replays]

OUTPATH = os.path.join('/', *current_path.split('/')[:-1], 'model', 'model.ckp')

paths = []
for path in _paths:
    paths += prepare_filepaths(path)
shuffle(paths)

assert len(paths) == 3242 # Total num of replays used during training

# Just take a split of the randomized replays for use in validation
valid_paths = paths[-200:]
paths = paths[:-200]


batch_size = 4
dims = 60 # We pad all frames to this value

def get_data(path):
    """
    Data loader for use in the multiprocessing worker. 
    
    We can perform additional preprocessing and augmentation in this function 
    virtually for free (when using a GPU), since we can prepare a batch of data 
    while the last batch is being computed.
    """
    
    data = np.load(path)
    replay = data['replay']
    moves = data['moves']
    
    shape = replay.shape[0]
    
    # Test for quality
    if np.sum(replay[-2, :, :, 1]>0) < (np.sum(replay[-2, :, :, 2]>0)):
        return None

    # Ensure these meet the spec
    if replay.shape[1] <= 50 and replay.shape[2] <= 50 and replay.shape[0] > 15:
    
        # Padding is identical so we don't need the second output
        all_frames, _frames_padding = pad_replay(replay, dims)
        all_moves, _ = pad_replay(np.expand_dims(moves, axis=-1), dims)
        
        all_frames = all_frames[:-1]
        
        all_moves = np.squeeze(all_moves)
        
        moves = np.array(all_moves)

        hold = moves == 0
        north = moves == 1
        east = moves == 2
        south = moves == 3
        west = moves == 4
        
        moves = np.array([hold, north, east, south, west])

        moves = np.rollaxis(moves, 0, 4)

        all_moves = moves.copy()

        j = random.choice(range(8))

        moves = []
        frames = []
        # Do rotation and mirror augmentation to expand the data 8X
        if j != 0:
            for frame, move in zip(all_frames, all_moves):
                move, frame, frames_padding = aug(move, frame, j,
                                                  _frames_padding)
                moves.append(move)
                frames.append(frame)
            all_moves = np.array(moves)
            all_frames = np.array(frames)
        else:
            all_moves = np.array(all_moves)
            all_frames = np.array(all_frames)
            frames_padding = _frames_padding

        return np.array(all_frames), np.array(all_moves), frames_padding
    else:
        return None


def condense(replay):
    """
    Previously I prepared the data to utilize 4 seperate channels, but using 3
    channels seemed to perform slightly better, so we need to reformat the
    data here to go from 4->3 channels.
    """
    replay[:, :, :, 0] += replay[:, :, :, 1]
    replay[:, :, :, 0] += replay[:, :, :, 2]
    
    mine = replay[:, :, :, 1]>0
    enemy = replay[:, :, :, 2]>0
    
    replay[:, :, :, 1] = mine.astype('float32')
    replay[:, :, :, 1] -= enemy.astype('float32')

    replay = np.delete(replay, 2, axis=3) # Remove the unneeded channel
    
    replay[:, :, :, 0] = replay[:, :, :, 0] - 0.5
    replay[:, :, :, 2] = replay[:, :, :, 2] - 0.5

    return replay

def worker(_queue, beg_frame, end_frame, _paths):

    data_buffer = []

    while True:
    
        # Get random replay
        try:
            path = random.choice(_paths)
        except:
            print(len(_paths))
            print(_paths[:5])
            continue
        
        data = get_data(path)
        
        if type(data) == type(None):
            continue
        
        replay, moves, padding = data

        # Normalize the data
        replay = replay/255.

        rewards = np.ones((replay.shape[0],))
        
        mask = replay[:, :, :, 1] > 0
        mask = mask.astype('float32')

        # Also account for padding
        mask[:, :padding[0][0], :] = 0
        mask[:, -padding[0][1]:, :] = 0
        mask[:, :, :padding[1][0]] = 0
        mask[:, :, -padding[1][1]:] = 0
        
        replay = condense(replay) # Convert 4->3 channels

        counter = 0
        for m, r, re, ma in zip(np.split(moves, moves.shape[0]),
                                    np.split(rewards,
                                            rewards.shape[0]),
                                    np.split(replay[:, :, :],
                                            replay[:, :, :].shape[0]),
                                    np.split(mask[:, :, :],
                                            mask[:, :, :].shape[0])):
            
            if counter >= beg_frame and counter <= end_frame:
                data_buffer.append((np.squeeze(m), np.squeeze(r),
                                    np.squeeze(re), np.squeeze(ma)))
            counter += 1
            
            if counter == end_frame:
                break

    
        if len(data_buffer) >= 2000:
            while len(data_buffer) > batch_size:
                shuffle(data_buffer)
                pre_batch = data_buffer[:batch_size]
                data_buffer = data_buffer[batch_size:]
                
                # Create the batch
                rewards = []
                frames = []
                moves = []
                masks = []
                
                for tup in pre_batch:
                    move = tup[0]
                    reward = tup[1]
                    frame = tup[2]
                    mask = tup[3]
                    
                    moves.append(move)
                    rewards.append(reward)
                    frames.append(frame)
                    masks.append(mask)
                
                _queue.put((np.array(moves), np.array(rewards),
                            np.array(frames), np.array(masks)))

# Prepare multiple processes to offload computation while training on GPU
manager = multiprocessing.Manager()
beg_queue = manager.Queue(100)
mid_queue = manager.Queue(100)
end_queue = manager.Queue(100)

multiprocessing.Pool(1, worker, (beg_queue, 0, 25, paths))
multiprocessing.Pool(1, worker, (mid_queue, 26, 125, paths))
multiprocessing.Pool(1, worker, (end_queue, 126, 999, paths))

beg_queue_v = manager.Queue(30)
mid_queue_v = manager.Queue(30)
end_queue_v = manager.Queue(30)

multiprocessing.Pool(1, worker, (beg_queue_v, 0, 25, valid_paths))
multiprocessing.Pool(1, worker, (mid_queue_v, 26, 125, valid_paths))
multiprocessing.Pool(1, worker, (end_queue_v, 126, 999, valid_paths))


observation_placeholder = tf.placeholder("float32", [batch_size, dims, dims, 3])
moves_placeholder = tf.placeholder("float32", [batch_size, dims, dims, 5])
mask_placeholder = tf.placeholder("float32", [batch_size, dims, dims])
rewards_placeholder = tf.placeholder("float32", [batch_size, 1])
training_placeholder = tf.placeholder(tf.bool)
learning_rate = tf.Variable(1e-4, trainable=False)

with tf.variable_scope('model'):
    h_conv1 = convolve(observation_placeholder, 3, 256, 'c1')
    h_conv2 = convolve(h_conv1, 256, 128, 'c2', kernel=3)
    h_conv3 = convolve(h_conv2, 128, 128, 'c3', kernel=3)
    h_conv5 = convolve(h_conv3, 128, 128, 'c51', kernel=3)
    h_conv7 = convolve(h_conv5, 128, 128, 'c61', kernel=3)
    h_conv4 = convolve(h_conv7, 128, 256, 'c5', kernel=3)
    h_conv18 = convolve(h_conv4, 256, 5, 'c18', False, kernel=5)


prediction_layer = h_conv18

ce = tf.nn.softmax_cross_entropy_with_logits(prediction_layer, moves_placeholder) * mask_placeholder

pre_loss = tf.reduce_sum(tf.reshape(ce, [batch_size, -1]) * rewards_placeholder)

loss = pre_loss/tf.reduce_sum(tf.cast(mask_placeholder>0, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    avg = []
    best_score = 1000

    for step in range(1000000):
        
        # Sample from one of the queues
        which_queue = step % 3
        if which_queue == 0:
            data = beg_queue.get()
            reward_scale = 1.
        elif which_queue == 1:
            if not mid_queue.qsize():
                continue
            data = mid_queue.get()
            reward_scale = 0.75
        elif which_queue == 2:
            if not end_queue.qsize():
                continue
            data = end_queue.get()
            reward_scale = 0.45
        else:
            Error

        moves, rewards, frames, masks = data


        masks = masks.astype('float32')
        zeros = moves[:, :, :, 0]
        masks = masks - 0.5281*zeros.astype('float32') # How much to decrease
        masks = masks * (masks>0)
        masks = masks.astype('float32')
        
        l, _ = sess.run([loss, optimizer],
                           feed_dict={
                            observation_placeholder: frames,
                            rewards_placeholder: np.expand_dims(rewards, 1)*reward_scale,
                            moves_placeholder: moves,
                            mask_placeholder: masks,
                            training_placeholder: True})

        if step == 140000:
            print('Lowered learning rate')
            sess.run(learning_rate.assign(3e-5))
        if step == 180000:
            print('Lowered learning rate')
            sess.run(learning_rate.assign(1e-5))
        if step == 220000:
            print('Lowered learning rate')
            sess.run(learning_rate.assign(1e-6))
        avg.append(l/reward_scale)
        if step % 1000 == 0:
            print('***')
        if step % 10000 == 0 or step == 1000:
            
            print('Step {0} T: {1}'.format(step, np.mean(avg)*100))
            avg = []
            
            bloss = []
            mloss = []
            eloss = []
            for vstep in range(150):
                moves, rewards, frames, masks = beg_queue_v.get()
                l, cl, pl = sess.run([loss, ce, pre_loss],
                                   feed_dict={
                                    observation_placeholder: frames,
                                    rewards_placeholder: np.expand_dims(rewards, 1),
                                    moves_placeholder: moves,
                                    mask_placeholder: masks,
                                    training_placeholder: False})
                bloss.append(l)
            print('Step {0} B: {1}'.format(step, np.mean(bloss)*100))
            
            for vstep in range(100):
                moves, rewards, frames, masks = mid_queue_v.get()
                l, cl, pl = sess.run([loss, ce, pre_loss],
                                   feed_dict={
                                    observation_placeholder: frames,
                                    rewards_placeholder: np.expand_dims(rewards, 1),
                                    moves_placeholder: moves,
                                    mask_placeholder: masks,
                                    training_placeholder: False})
                mloss.append(l)
            print('Step {0} M: {1}'.format(step, np.mean(mloss)*100))
                
            for vstep in range(75):
                moves, rewards, frames, masks = end_queue_v.get()
                l, cl, pl = sess.run([loss, ce, pre_loss],
                                   feed_dict={
                                    observation_placeholder: frames,
                                    rewards_placeholder: np.expand_dims(rewards, 1),
                                    moves_placeholder: moves,
                                    mask_placeholder: masks,
                                    training_placeholder: False})
                eloss.append(l)
            print('Step {0} E: {1}'.format(step, np.mean(eloss)*100))
            new_score = np.mean(bloss)*100 + np.mean(mloss)*30 + np.mean(eloss)*5
            if new_score < best_score:
                saver.save(sess, OUTPATH)
                best_score = new_score
                print('New best score: {}'.format(best_score))






