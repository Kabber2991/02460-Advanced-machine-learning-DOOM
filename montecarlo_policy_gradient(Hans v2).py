# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 19:11:43 2019
@author: Sasha
"""
#from IPython.display import HTML
#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/wLTQRuizVyE?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform # Help us to preprocess the frames
from scipy import signal

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')



"""
Step 2
Create our environment
"""
def create_environment():
    game = DoomGame()
    # Load the correct configuration
    game.load_config("//Users//HansLindenbergStentoft//anaconda3//lib//python3.6//site-packages//vizdoom//scenarios//defend_the_center.cfg")
    
    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path("//Users//HansLindenbergStentoft//anaconda3//lib//python3.6//site-packages//vizdoom//scenarios//defend_the_center.wad")
    
    game.init()
    
    # Here our possible actions
    # [[1,0,0],[0,1,0],[0,0,1]]
    possible_actions  = np.identity(3,dtype=int).tolist()
    
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
action_size = game.get_available_buttons_size() # 3 possible actions: turn left, turn right, shoot
stack_size = 4 # Defines how many frames are stacked together

## TRAINING HYPERPARAMETERS
learning_rate = 0.0003 
num_epochs = 100  # Total epochs for training 1000

batch_size = 1000 # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU 1000
gamma = 0.99 # Discounting rate

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
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
                self.logits = tf.layers.dense(inputs = self.fc, 
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units = 3, 
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
                
# Reset the graph
tf.reset_default_graph()


# Instantiate the PGNetwork
PGNetwork = PGNetwork(state_size, action_size, learning_rate)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



""" 
Step 6
SET UP TENSORBOARD
"""

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard//policy_gradients/1")

## Losses
tf.summary.scalar("Loss", PGNetwork.loss)

## Reward mean
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_ )

write_op = tf.summary.merge_all()


def rewardfunction(reward,action,ammo,health,d_ammo,d_health):
    #if the agent shoots and misses
    if action[2]==1 and reward==0:
        reward+=-1
    #If it turns right7left and shoots and kills
    elif action[0]==1 or action[1]==1 and action[2]==1 and reward==1:
        reward+=2
    #if agent makes a kill
    elif reward==1:
        reward+=20
    #if agent dies
    elif reward==-1:
        reward+=-200
    #if the agent looses life. When a monster is close we want it to turn around and kill it. 
    # So if it's getting hurt and not moving put shoots and miss then penelize
    #elif d_health!=0 and action[0]==0 or action[1]==0 and action[2]==1 and reward==0 :   
     #   reward+=-8
        
    return reward

"""
Step 7
TRAIN OUR AGENT
"""

def make_batch(batch_size, stacked_frames):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards, = [], [], [], [], []
    
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num  = 1
    Kil=[]
    Kills= 0
    OverAllHighestKills=[]
    
    # Launch a new episode
    game.new_episode()
        
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    
    #state of variables at start
    ammo=game.get_state().game_variables[0]
    health=game.get_state().game_variables[1] #index 1 for defendcenter
    while True:
        # Run State Through Policy & Calculate Action
        try:
            action_probability_distribution = sess.run(PGNetwork.action_distribution, 
                                                   feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        except ValueError:
            pass
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        #30% chance that we take action a2)
        try:
            action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        except ValueError:
            pass
        action = possible_actions[action]
        
        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()
        
        if reward==1:
            Kills +=1
            print("EPIC KILL")
        elif reward==-1:
            Kil.append(Kills)
            print("YOU DIED")
            Kills=0
        
        if game.get_state()!=None:
            #Prints ammo and health
            #change in variables per frame
            d_ammo=ammo-game.get_state().game_variables[0]
            d_health=health=health-game.get_state().game_variables[1] #index 1 for defendcenter
            
            #state of variables per frame
            ammo=game.get_state().game_variables[0]
            health=game.get_state().game_variables[1] #index 1 for defendcenter
            
        reward=rewardfunction(reward,action,ammo,health,d_ammo,d_health)

        OverAllHighestKills.append(Kil)
        #print(OverAllHighestKills)
    
        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        #Kills.append(Kil)
        
        if done:
            # The episode ends so no next state
            next_state = np.zeros((3,100, 180), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            

            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
           
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
                
            # Reset the transition stores
            rewards_of_episode = []
            
            # Add episode
            episode_num += 1
            
            # Start a new episode
            game.new_episode()

            # First we need a state
            state = game.get_state().screen_buffer

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
         
        else:
            # If not done, the next_state become the current state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
                         
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num, np.max(OverAllHighestKills)



# Keep track of all rewards total for each batch
allRewards = []

total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []
Highest_total_kills= 0
Highest_total_kills_overall= []

# Saver
saver = tf.train.Saver()

if training == True:
    # Load the model
    #saver.restore(sess, "./models/model.ckpt")

    while epoch < num_epochs + 1:
        # Gather training data
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb, OverAllHighestKills = make_batch(batch_size, stacked_frames)

        ### These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)

        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)
        mean_reward_total.append(mean_reward_of_that_batch)

        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

        # Calculate maximum reward recorded 
        maximumRewardRecorded = np.amax(allRewards)
        
        #Calculate maximum Kills
        #maxkils = np.amax(OverAllHighestKills)
        Highest_total_kills_overall.append(OverAllHighestKills)
        maxkils1 = np.amax(Highest_total_kills_overall)

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))
        print("Max kills so far: {}".format(maxkils1))

        # Feedforward, gradient and backpropagation
        loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 100,160,4)),
                                                            PGNetwork.actions: actions_mb,
                                                                     PGNetwork.discounted_episode_rewards_: discounted_rewards_mb 
                                                                    })

        #print("Training Loss: {}".format(loss_))

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 100,160,4)),
                                                            PGNetwork.actions: actions_mb,
                                                                     PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                                                    PGNetwork.mean_reward_: mean_reward_of_that_batch
                                                                    })

        #summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
        writer.add_summary(summary, epoch)
        writer.flush()

        # Save Model
        if epoch % 10 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")
        epoch += 1
        
"""
Step 8
WATCH THE AGENT PLAYING
"""

# Saver
saver = tf.train.Saver()

with tf.Session() as sess:
    print("Starting training")
    game = DoomGame()
    # Load the correct configuration 
    game.load_config("//Users//HansLindenbergStentoft//anaconda3//lib//python3.6//site-packages//vizdoom//scenarios//defend_the_center.cfg")
    
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("//Users//HansLindenbergStentoft//anaconda3//lib//python3.6//site-packages//vizdoom//scenarios//defend_the_center.wad")
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    
    game.init()
    
    for i in range(10):
        # Launch a new episode
        game.new_episode()

        # Get a new state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        
        
        while not game.is_episode_finished():
        
            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(PGNetwork.action_distribution, 
                                                       feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})

            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            #30% chance that we take action a2)
            action = np.random.choice(range(action_probability_distribution.shape[1]), 
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = possible_actions[action]

            # Perform action
            reward = game.make_action(action)
            done = game.is_episode_finished()
            

            if done:
                break
            else:
                # If not done, the next_state become the current state
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        

        print("Score for episode ", i, " :", game.get_total_reward())
    game.close()