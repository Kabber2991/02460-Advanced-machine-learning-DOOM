"""
Created on Sun Mar 24 19:11:43 2019

@author: Sasha
"""

"""
NOTICE
NOTICE
NOTICE
NOTICE

EXAMPLE:
This Policy gradiant is based on the example given in the following article:
https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f

TEAM:
    S124217
    s145436
    s174454
    
In order to be as transparent with what is coded by the team, 
and what was taken from the example. The script will be seperated in sections
where the main contributor, that being team or example is marked in the begining 
"""

#from IPython.display import HTML
#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/wLTQRuizVyE?showinfo=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import vizdoom
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

#%% TEAM
with tf.device('/device:GPU:1'):
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
    
#%% EXAMPLE 
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
        game.set_living_reward(+1)
        game.set_console_enabled(True)
        
        game.init()
#%% TEAM    
        # Here our possible actions
        # [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
        possible_actions=[[0,0,0]]
        possible_actions.extend(np.identity(3,dtype=int).tolist())

#%% EXAMPLE    
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
        cropped_frame = frame[:,50:,:]
    
        # Normalize Pixel Values
        normalized_frame = cropped_frame/255.0

#%% TEAM    
        #Ændrer RGB til greyscale
        greyscale = np.array(normalized_frame).mean(axis=0)
    
        # Resize
        #preprocessed_frame = transform.resize(normalized_frame, [100,160])
        #Sampler screen
        downscale_greyscale=signal.resample(greyscale,num=80,axis=0)
        preprocessed_frame=signal.resample(downscale_greyscale,num=90,axis=1)
        return preprocessed_frame
    
    
    stack_size = 4 # We stack 4 frames
    
#%% EXAMPLE
    
    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((80,90), dtype=np.int) for i in range(stack_size)], maxlen=4)
    
    def stack_frames(stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = preprocess_frame(state)
    
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros((80,90), dtype=np.int) for i in range(stack_size)], maxlen=4)
    
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
    

#%% TEAM   
    """
    Step 4
    SET HYPERPARAMETERS
    """
    
    ### ENVIRONMENT HYPERPARAMETERS
    state_size = [80,90,4] # Our input is a stack of 4 frames hence 100x160x4 (Width, height, channels)
    action_size = len(possible_actions)# 4 possible actions: nothing, turn left, turn right, shoot
    stack_size = 4 # Defines how many frames are stacked together
    
    ## TRAINING HYPERPARAMETERS
    learning_rate = 0.0002
    num_epochs = 1000  # Total epochs for training
    
    batch_size = 1000 # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
    gamma = 0.99 # Discounting rate
    
    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True
    
#%% EXAMPLE
    
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
                    self.frame_surv_ = tf.placeholder(tf.float32, name = "frame_surv")
    
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
                                                 activation = tf.nn.elu,
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
                                         activation = tf.nn.elu,
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
                                         activation = tf.nn.elu,
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
                                          units = 512,                       #CHANGED THIS FROM 512! TRY TO SOLVE 00M ResourcheExhaustError
                                          activation = tf.nn.elu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                               
                                        name="fc1")
    
                with tf.name_scope("logits"):
                    #units er antal af mulige actions
                    self.logits = tf.layers.dense(inputs = self.fc,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                  units = 4,
                                                activation=None)

#%% TEAM                   
                with tf.name_scope("value"):
                    self.value = tf.layers.dense(inputs=self.fc, 
                                                 units = 1)

                with tf.name_scope("softmax"):
                    self.action_distribution = tf.nn.softmax(self.logits)
    
    
                with tf.name_scope("loss"):
                    # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                    # If you have single-class labels, where an object can only belong to one class, you might now consider using
                    # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                    self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.logits, labels = self.actions)
                    self.pg_loss = tf.reduce_mean((self.discounted_episode_rewards_ - self.value) *self.neg_log_prob)
                    
                    self.val_loss = tf.reduce_mean(tf.square(self.discounted_episode_rewards_ - self.value))
                    self.loss = 0.7*self.pg_loss + self.val_loss 
                        
               #pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
               #value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))



                    #self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

    
                with tf.name_scope("train"):
                    self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)





# =============================================================================
#                 with tf.name_scope("train"):
#                     
#                     
#                     # Create Optimizer
#                     self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
#                     self.grads = tf.gradients(self.loss, tf.trainable_variables())
#                     self.grads, _ = tf.clip_by_global_norm(self.grads, 40) # gradient clipping
#                     self.grads_and_vars = list(zip(self.grads, tf.trainable_variables()))
#                     self.train_opt = self.optimizer.apply_gradients(self.grads_and_vars)
#  
# =============================================================================
                    #self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    
              #  with tf.name_scope("train"):
                    #self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
                   # self.train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

#%% EXAMPLE 
                    
    # Reset the graph
    tf.reset_default_graph()
    
    # Instantiate the PGNetwork
    PGNetwork = PGNetwork(state_size, action_size, learning_rate)
    
    # Initialize Session
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    
#%% TEAM    
    """
    Step 6
    SET UP TENSORBOARD
    """
    
    # Setup TensorBoard Writer
    NAME_pg = "policy_grad{}".format(int(time.time()))
    writer = tf.summary.FileWriter("/tensorboard/policy_gradients/health_gathering/{}".format("07polic_test2")) #New saver in different folders format(NAME_pg)
    
    writer.add_graph(sess.graph)
    
    ## Losses
    tf.summary.scalar("Loss", PGNetwork.loss)
    
    ## Reward mean
    tf.summary.scalar("Perf/Reward", PGNetwork.mean_reward_ )
    
    tf.summary.scalar("frame_surv", PGNetwork.frame_surv_ )
    
    write_op = tf.summary.merge_all()
    
    """
    def rewardfunction(reward,action,ammo,health,d_ammo,d_health):
        #if the agent shoots and misses
        #if agent makes a kill
        if reward==1:
            reward+=6
        #if agent dies
    # =============================================================================
        elif reward==1:
             reward+=6
    
        #if the agent looses life
        elif d_health!=0:
            reward+=-6
    
        return reward
    
    def rewardfunction(reward,action,ammo,health,d_ammo,d_health):
        #if the agent shoots and misses
        if action[2]==1 and reward==0:
            reward+=-2
        #if agent makes a kill
        elif reward==1:
            reward+=6
        #if agent dies
        elif reward==-1:
            reward+=-100
        #if the agent looses life
        elif d_health!=0:
            reward+=-6
    
        return reward
    """

#%% EXAMPLE 
    
    """
    Step 7
    TRAIN OUR AGENT
    """
    tf.trainable_variables()
    
    def make_batch(batch_size, stacked_frames):
        # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
        states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    
        # Reward of batch is also a trick to keep track of how many timestep we made.
        # We use to to verify at the end of each episode if > batch_size or not.
    
        # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
        episode_num  = 1
        frames_survived=0
#        Kills= 0
#        Kill_list=[]    
        
        # Launch a new episode
        game.new_episode()
    
        # Get a new state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
        #state of variables at start
#        ammo=game.get_state().game_variables[0]
        health=game.get_state().game_variables[0]   #changed from 1 to 0
        
        while True:
            # Run State Through Policy & Calculate Action
            try:
                action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                       feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
            except ValueError:
                continue
            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            #30% chance that we take action a2)
            try:
                action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
    
            except ValueError:
                #pass
                continue
            action = possible_actions[action]
            reward = game.make_action(action)
            done = game.is_episode_finished()
    
            frames_survived+=1
    
#            if reward==1:
#                Kills +=1
#                print("EPIC KILL")
#            elif reward==-1:
#                print("YOU DIED")
    
            if game.get_state()!=None:
                #Prints ammo and health
                #change in variables per frame
#                d_ammo=ammo-game.get_state().game_variables[0]
                d_health=health=health-game.get_state().game_variables[0]      #changed from 1 to 0
    
                #state of variables per frame
#                ammo=game.get_state().game_variables[0]
                health=game.get_state().game_variables[0]                     #changed from 1 to 0
    
           # reward=rewardfunction(reward,action,ammo,health,d_ammo,d_health)
    
            # Store results
            states.append(state)
    
            #makes sure that the none action is added to the acion list
            temp=[1-sum(action)]
    
            temp.extend(action)
            actions.append(temp)
    
            rewards_of_episode.append(reward)
    
            if done:
                #appends kills
#                Kill_list.append(Kills)
#                Kills=0
                
                # The episode ends so no next state
                next_state = np.zeros((3,80, 90), dtype=np.int)
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
            
        return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num,frames_survived  #,Kill_list
    
#%% TEAM
    # Keep track of all rewards total for each batch
    allRewards = []
    
    total_rewards = 0
    maximumRewardRecorded = 0
    mean_reward_total = []
    epoch = 1
    average_reward = []
#    Highest_total_kills= 0
#    Highest_total_kills_overall= []
    
    # Saver
    saver = tf.train.Saver()
    
    #Specify if and which model to load on resuming training
    reload=False
    reloadModel="health_wagon_test.ckpt"
    
    #Specify the new name for the model if the script is to make a new model from scratch
    modelname="line_wagon_final.ckpt"
    
    
    if training == True:
        # Load the model
        if reload==True:
            restorepath= "./models/" + reloadModel
            saver.restore(sess, "./models/" + reloadModel)
            print("Resuming training of model: " + reloadModel)
   
#%% EXAMPLE
        while epoch < num_epochs + 1:
            # Gather training data
            states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb,frames_mb = make_batch(batch_size, stacked_frames)
    
            ### These part is used for analytics
            # Calculate the total reward ot the batch
            total_reward_of_that_batch = np.sum(rewards_of_batch)
            allRewards.append(total_reward_of_that_batch)
    
            # Calculate the mean reward of the batch
            # Total rewards of batch / nb episodes in that batch
            mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
            mean_reward_total.append(mean_reward_of_that_batch)
    
            # Calculate the average reward of all training
            # mean_reward_of_that_batch / epoch
            average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)
    
            # Calculate maximum reward recorded
            maximumRewardRecorded = np.amax(allRewards)
            
            #Calculate maximum Kills
            #maxkils = np.amax(OverAllHighestKills)
#            Highest_total_kills_overall.extend(kills)
#            maxkils1 = np.amax(Highest_total_kills_overall)
            
    
            print("==========================================")
            print("Epoch: ", epoch, "/", num_epochs)
            print("-----------")
            print("Number of training episodes: {}".format(nb_episodes_mb))
            print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
            print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
            print("Average Reward of all training: {}".format(average_reward_of_all_training))
            print("Max reward for a batch so far: {}".format(maximumRewardRecorded))
            print("Frames survived: {}".format(frames_mb))

#            print("Max kills so far: {}".format(maxkils1))
            
            # Feedforward, gradient and backpropagation
            loss_, _, neg_log_prob_, disc_ep_rew = sess.run([PGNetwork.loss, PGNetwork.train_opt, PGNetwork.neg_log_prob, PGNetwork.discounted_episode_rewards_], feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 80,90,4)),
                                                                PGNetwork.actions: actions_mb,
                                                                         PGNetwork.discounted_episode_rewards_: discounted_rewards_mb
                                                                        })
          #  buttons = game.get_available_buttons()
          #  print(buttons)
            print("Training Loss: {}".format(loss_))

#%% TEAM            
            
            #Renders a state and calculates the probabilities for each action
            if epoch % 20 == 0:
                game.new_episode()
                #game.get_state().screen_buffer
                stat = preprocess_frame(game.get_state().screen_buffer)
                prob, val = sess.run([PGNetwork.action_distribution, PGNetwork.value],
                                                           feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 80,90,4))})


                plt.imshow(stat, interpolation='nearest')
                plt.show()
                print('Turn Left: {:4.2f}  Turn Right: {:4.2f}  Move Forward {:4.2f} Do nothing {:4.2f}'.format(prob[0][0],prob[0][2], prob[0][1], prob[0][3]))
                print('Approximated State Value: {:4.4f}'.format(val[0][0]))
                
            #print("-------")
            #print("-------")
            #print("NEG_LOG_PROB: {} {}".format(len(neg_log_prob_),neg_log_prob_))
            #print("DISC_EP_REW: {} {}".format(len(disc_ep_rew),disc_ep_rew))
            # Write TF Summaries
            
#%% EXAMPLE
            summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 80,90,4)),
                                                                PGNetwork.actions: actions_mb,
                                                                         PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                                                        PGNetwork.mean_reward_: mean_reward_of_that_batch,
                                                                        PGNetwork.frame_surv_:frames_mb
                                                                        })
    
            writer.add_summary(summary, epoch)
            writer.flush()
    
            # Save Model
            #if epoch % 10 == 0:
    
            if epoch % 10 ==0:
                #Ensures that the reloaded model is overwritten with the new model
                if reload==True:
                    saver.save(sess,restorepath)
                    print("Resumed model saved")
                #Ensure that a new model is created if so specified
                else:
                    saver.save(sess, "./models/"+ modelname)
                    print("New model saved")
    
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
        game.load_config(scenario + DOOM_SETTINGS[Select_level][0])
    
        # Load the correct scenario (in our case basic scenario)
        game.set_doom_scenario_path(scenario + DOOM_SETTINGS[Select_level][1])
    
        # Load the model
        if reload==True:
            saver.restore(sess,restorepath)
        else:
            saver.restore(sess, "./models/"+ modelname)
    
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
            game.new_episode()
            #game.get_state().screen_buffer
            stat = preprocess_frame(game.get_state().screen_buffer)
            prob, val = sess.run([PGNetwork.action_distribution, PGNetwork.value],
                                                       feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 80,90,4))})
#%% TEAM
            plt.imshow(stat, interpolation='nearest')
            plt.show()
            print('Turn Right: {:4.2f}  Turn Left: {:4.2f}  Move Forward {:4.2f} Do nothing {:4.2f}'.format(prob[0][0],prob[0][2], prob[0][1], prob[0][3]))
            print('Approximated State Value: {:4.4f}'.format(val[0][0]))
            
        game.close()
