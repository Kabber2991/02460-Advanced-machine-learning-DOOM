# -*- coding: utf-8 -*-
"""
NOTICE
NOTICE
NOTICE
NOTICE

EXAMPLE:
This A3C is based on the example given in the following GitHub:
https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

TEAM:
    S124217
    s145436
    s174454
    
In order to be as transparent with what is coded by the team, 
and what was taken from the example. The script will be seperated in sections
where the main contributor, that being team or example is marked in the begning 
    
"""
#%% General settings
import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from helper import *
import vizdoom
from vizdoom import *
import os

from random import choice
from time import sleep
from time import time
import datetime

# Constants
NUM_ACTIONS = 43
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

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
Select_level = 4

dp = os.path.dirname(vizdoom.__file__)
scenario = dp + "/scenarios/"

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #%% neural network  TEAM
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=32,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=64,
                kernel_size=[4,4],stride=[2,2],padding='VALID')

# This fourth layer seems to slow the A3C too much, and therefore it is left out
#            self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
#                inputs=self.conv2,num_outputs=128,
#                kernel_size=[4,4],stride=[2,2],padding='VALID')            
            
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            
            #%% EXAMPLE
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            
            self.state_in = (c_in, h_in)
            
            rnn_in = tf.expand_dims(hidden, [0])
            
            step_size = tf.shape(self.imageIn)[:1]
            
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            
            # setting up the dynamic rnn to explore tempero
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            
            lstm_c, lstm_h = lstm_state
            
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                
                #%% loss function - TEAM
                # new entropy calculation
                # taken from:
                # https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
                # setting up entropy
                self.policy_clipped = tf.clip_by_value(self.policy, 1e-10, 0.9999999)
                self.entropy = -tf.reduce_mean(tf.reduce_sum(self.actions_onehot * tf.log(self.policy_clipped)+ (1 - self.actions_onehot) * tf.log(1 - self.policy_clipped), axis=1))
                
                # old entropy calculation
                #self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                
                # Total loss function
                # Ensuring and entropy doesn't end up making a single bad decision
                # by controlling the weight of each factor
                # ensuring that the policy and value is weighted equally in the loss function
                self.loss = 0.1*self.value_loss + self.policy_loss + self.entropy *0.00001
                
                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                #%% example
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                
class Worker():
    def __init__(self,game,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_kills=[]
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(model_path +"//train_"+str(self.number))
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        #The Below code is related to setting up the Doom environment
        game.set_doom_scenario_path(scenario + DOOM_SETTINGS[Select_level][1]) #This corresponds to the simple task we will pose our agent
        game.set_doom_map("map01")
        
        # setter skærm opløsningen
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        
        # render options
        game.set_render_hud(False)
        game.set_render_crosshair(False)
        game.set_render_weapon(True)
        game.set_render_decals(False)
        game.set_render_particles(False)
        
        #Loader available buttions
        # settings for defend the center:
        #%% TEAM
        if Select_level==2:
            game.add_available_button(Button.TURN_LEFT)
            game.add_available_button(Button.TURN_RIGHT)
            game.add_available_button(Button.ATTACK)
            
            game.set_living_reward(0)
            
            # loader game variables
            game.add_available_game_variable(GameVariable.AMMO2)
            game.add_available_game_variable(GameVariable.HEALTH)
            game.add_available_game_variable(GameVariable.POSITION_X)
            game.add_available_game_variable(GameVariable.POSITION_Y)
        # setting for defend the line:
        elif Select_level==3:
            game.add_available_button(Button.TURN_LEFT)
            game.add_available_button(Button.TURN_RIGHT)
            game.add_available_button(Button.ATTACK)
            
            game.set_living_reward(0)
            
            # loader game variables
            game.add_available_game_variable(GameVariable.AMMO2)
            game.add_available_game_variable(GameVariable.HEALTH)
            game.add_available_game_variable(GameVariable.POSITION_X)
            game.add_available_game_variable(GameVariable.POSITION_Y)
        # setting for health gathering
        elif Select_level==4:
            game.add_available_button(Button.TURN_LEFT)
            game.add_available_button(Button.TURN_RIGHT)
            game.add_available_button(Button.MOVE_FORWARD)
            game.set_living_reward(1)

            # loader game variables
            game.add_available_game_variable(GameVariable.HEALTH)
            game.add_available_game_variable(GameVariable.POSITION_X)
            game.add_available_game_variable(GameVariable.POSITION_Y)
        
        #%% example
        #setter start time
        game.set_episode_start_time(10)
        game.set_window_visible(True)
        game.set_sound_enabled(False)
        game.set_console_enabled(True)
        
        # reward setting for game
        game.set_living_reward(0)
        
        game.set_mode(Mode.PLAYER)
        game.init()
        
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        #End Doom set-up
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        self.summary_writer.add_graph(sess.graph)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_kill = 0
                episode_step_count = 0
                d = False
                
                #%% reward function - TEAM
                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                
                    #state of variables at start
                if Select_level!=4:
                    ammo=self.env.get_state().game_variables[0]
                    health=self.env.get_state().game_variables[1]
                else:
                    ammo=0
                    health=self.env.get_state().game_variables[0]
                    
                while self.env.is_episode_finished() == False:
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    # if r = 0 then kill confirmed
                    r = self.env.make_action(self.actions[a])
                    
                    # reward function
                    if self.env.get_state()!=None:
                        #Prints ammo and health
                        #change in variables per frame
                        if Select_level!=4:
                            d_ammo=ammo-self.env.get_state().game_variables[0]
                            d_health=health=health-self.env.get_state().game_variables[1]
                
                            #state of variables per frame
                            ammo=self.env.get_state().game_variables[0]
                            health=self.env.get_state().game_variables[1]
                        else:
                            d_ammo=0
                            d_health=health=health-self.env.get_state().game_variables[0]
                
                            #state of variables per frame
                            ammo=0
                            health=self.env.get_state().game_variables[0]
                            
                    #If health is gained
                    if d_health>0:
                        r+=10
                    if d_health<0:
                        r-=0.1
                    # If show is fired but no one is killed
                    if d_ammo!=0 and r!=1:
                        r-=1
                    # if a monster is killed
                    if r==1 and Select_level!=4:
                        r+=10
                        episode_kill+=1
                    
                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])
                    
                    # Adds 1 to the reward to ensure positivity
                    episode_reward += (r/100)+1
                    
                    #%% example
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == max_episode_length and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                self.episode_kills.append(episode_kill)
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
    
                # Saves summary statistics each time the network is trained
                if len(episode_buffer) != 0 and episode_count != 0:
                    if episode_count % 5 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model.cptk')
                        print ("Saved Model")
                        
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    mean_kills= np.mean(self.episode_kills[-5:])
                    
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Kills', simple_value=float(mean_kills))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                
                episode_count += 1
                #Ends coordinator
                if episode_count>1000:
                    coord.request_stop()

#%% Parameters - TEAM
max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 3 # Agent can move Left, Right, or Fire
learn_rate=1e-4 # Learning rate
epsi=0.1 # epsilon for Adam optimizer
#Specify if and which model to load on resuming training
load_model = True
model_path = './model_A3C_15052019_Ver10_' + str(Select_level)

tf.reset_default_graph()

if load_model==False:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
else:
    if not os.path.exists(model_path):
        print("Requested model doesn't exist")
        quit()

#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

#%% setting up of threads - EXAMPLE
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=learn_rate,epsilon=epsi)
    master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(DoomGame(),i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    
    