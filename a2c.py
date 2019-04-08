# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:53:44 2019

@author: mc_ka
"""

#from IPython.display import HTML
#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/wLTQRuizVyE?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
import datetime
from scipy import signal
import os


from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

NUM_ACTIONS = 43

startdate=datetime.datetime.now()



DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                               # 0  - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],  # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],              # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                  # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],         # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],      # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                          # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, [x for x in range(NUM_ACTIONS) if x != 33], 0, 20] # 8 - Deathmatch
]
Select_level = 2



dp = os.path.dirname(vizdoom.__file__)
scenario = dp + "/scenarios/"


"""
Step 2
Create our environment
"""
def create_environment():
    game = DoomGame()
    
    
    # Load the correct configuration
    game.load_config(scenario + DOOM_SETTINGS[Select_level][0])
    
    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path(scenario + DOOM_SETTINGS[Select_level][1])
    game.init()

    # Here our possible actions
    # [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    possible_actions=[[0,0,0]]
    possible_actions.extend(np.identity(3,dtype=int).tolist())

    return game, possible_actions

game, possible_actions = create_environment()

"""
Step 3
Image processing
"""

"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|

        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.

    return preprocessed_frame

    """
def preprocess_frame(frame):
    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[:,40:,:]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0

    #Ændrer RGB til greyscale
    greyscale = np.array(normalized_frame).mean(axis=0)

    # Resize
    #preprocessed_frame = transform.resize(normalized_frame, [100,160])
    #Sampler screen
    downscale_greyscale=signal.resample(greyscale,num=100,axis=0)
    preprocessed_frame=signal.resample(downscale_greyscale,num=160,axis=1)

    return preprocessed_frame


stack_size = 4 # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((100,160), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,160), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards

"""
Step 4
SET HYPERPARAMETERS
"""

### ENVIRONMENT HYPERPARAMETERS
state_size = [100,160,4] # Our input is a stack of 4 frames hence 100x160x4 (Width, height, channels)
action_size = len(possible_actions)# 4 possible actions: nothing, turn left, turn right, shoot
stack_size = 4 # Defines how many frames are stacked together

## TRAINING HYPERPARAMETERS
learning_rate = 0.002
num_epochs = 1000  # Total epochs for training

batch_size = 1000 # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
gamma = 0.99 # Discounting rate

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

class ActorNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='Actornetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_= tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")


                # Add this placeholder for having this variable in tensorboard
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("conv1"):
                """
                First convnet:
                CNN
                BatchNormalization
                ELU
                """
                # Input is 84x84x4
                self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                             filters = 32,
                                             kernel_size = [8,8],
                                             strides = [4,4],
                                             padding = "VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                             name = "conv1")

                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm1')

                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                ## --> [20, 20, 32]

            with tf.name_scope("conv2"):
                """
                Second convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                     filters = 64,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv2")

                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm2')

                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
                ## --> [9, 9, 64]

            with tf.name_scope("conv3"):
                """
                Third convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                     filters = 128,
                                     kernel_size = [4,4],
                                     strides = [2,2],
                                     padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                     name = "conv3")

                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                       training = True,
                                                       epsilon = 1e-5,
                                                         name = 'batch_norm3')

                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                ## --> [3, 3, 128]

            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]

            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs = self.flatten,
                                      units = 512,
                                      activation = tf.nn.elu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    name="fc1")

            with tf.name_scope("logits"):
                #units er antal af mulige actions
                self.logits = tf.layers.dense(inputs = self.fc,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units = 4,
                                            activation=None)

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)


            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)


            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
