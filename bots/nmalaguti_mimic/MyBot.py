import tensorflow as tf
import numpy as np
import hlt
import os

from utils import pad_replay, dont_overconfidence, relu, convolve

max_dim = max(gameMap.height, gameMap.width)

dims = max(min(max_dim + max_dim%2 + 6, 52), 40)

observation_placeholder = tf.placeholder("float32", [None, None, None, 3])

with tf.variable_scope('model'):
    h_conv1 = convolve(observation_placeholder, 3, 256, 'c1')
    h_conv2 = convolve(h_conv1, 256, 128, 'c2', kernel=3)
    h_conv3 = convolve(h_conv2, 128, 128, 'c3', kernel=3)
    h_conv5 = convolve(h_conv3, 128, 128, 'c51', kernel=3)
    h_conv7 = convolve(h_conv5, 128, 128, 'c61', kernel=3)
    h_conv4 = convolve(h_conv7, 128, 256, 'c5', kernel=3)
    h_conv18 = convolve(h_conv4, 256, 5, 'c18', False, kernel=5)

action_distribution = h_conv18

saver = tf.train.Saver()

data = []
rewards = []

with tf.Session() as sess:
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'model/model.ckp'))
    
    frame = gameMap.initial_map
    
    my_row = np.where(np.any(frame[:, :, 1]>0, axis=1))[0]
    my_col = np.where(np.any(frame[:, :, 1]>0, axis=0))[0]
    
    center = frame.shape[0]//2, frame.shape[1]//2
    
    row_shift = int(center[0] - my_row)
    col_shift = int(center[1] - my_col)
    
    if row_shift < 0:
        row_shift = frame.shape[0] + row_shift
    
    if col_shift < 0:
        col_shift = frame.shape[1] + col_shift
    
    late_game = False

    hlt.send_init("nmalaguti_mimic")

    while True:

        moves = []
        orig_frame = gameMap.get_frame()
        

        old_shape = orig_frame.shape
        
        frame = np.roll(orig_frame, row_shift, axis=0)
        frame = np.roll(frame, col_shift, axis=1)
        
        frame = np.expand_dims(frame, axis=0)
        
        frame, padding = hlt.pad_replay(frame, dims)


        if not late_game:
        
            my_rows = np.where(np.any(frame[0, :, :, 1]>0, axis=1))[0]
            my_cols = np.where(np.any(frame[0, :, :, 1]>0, axis=0))[0]
            r_loc = int(min(my_rows)), int(max(my_rows))
            c_loc = int(min(my_cols)), int(max(my_cols))
            
            context_size = 11
            border_limit = (r_loc[1]+context_size - (r_loc[0]-context_size)) > 40 or (c_loc[1]+context_size - (c_loc[0]-context_size)) > 40
            
            if not border_limit and r_loc[0]-context_size >= 0 and r_loc[1]+context_size <= dims:
                frame = frame[:, r_loc[0]-context_size:r_loc[1]+context_size, :, :]
                r_start = r_loc[0]-context_size
                r_end = r_loc[1]+context_size
            else:
                r_start = 0
                r_end = dims
                late_game = True

            if not border_limit and c_loc[0]-context_size >= 0 and c_loc[1]+context_size <= dims:
                frame = frame[:, :, c_loc[0]-context_size:c_loc[1]+context_size, :]
                c_start = c_loc[0]-context_size
                c_end = c_loc[1]+context_size
            else:
                c_start = 0
                c_end = dims
                late_game = True

    
            has_padding = False
                
        else:
            r_start = 0
            r_end = dims
            c_start = 0
            c_end = dims
        
            has_padding = True
                
   
        actions = sess.run([action_distribution],
                           feed_dict={
                            observation_placeholder: frame})[0]

        actions = np.squeeze(actions)
        
        if has_padding:
            actions = actions[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], :]

        actions = actions.astype('float32')
        
        actions = np.argmax(actions, axis=2)
        
        if not has_padding:
            if not late_game:
                actions = dont_overconfidence(frame[0], actions)
            
            r1 = r_start - padding[0][0]
            r2 = dims-r_end + padding[0][1]
            r3 = c_start - padding[1][0]
            r4 = dims-c_end + padding[1][1]


            actions = np.pad(actions, ((r_start, dims-r_end),
                                        (c_start, dims-c_end)),
                                        mode='constant', constant_values=0)
            actions = actions[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]

        
        actions = np.roll(actions, actions.shape[0] - (row_shift), axis=0)
        actions = np.roll(actions, actions.shape[1] - (col_shift), axis=1)
        
        my_squares = orig_frame[:, :, 1] > 0
        
        for y in range(gameMap.height):
            for x in range(gameMap.width):
                if my_squares[y, x]:
                    move = hlt.Square(x, y, myID, None, None)
                    moves.append(hlt.Move(move, actions[y][x]))

    
        hlt.send_frame2(moves)



