#CS4287 Assignment 3
#Reinforcement Learning using DQN applied to Atari (Space Invaders!)

#Dependencies/Imports

#Calls tensor flow as well as the gym keras reenforcement learning assets we'll need for our atari enviroments
!pip install tensorflow==2.7 gym keras-rl2 gym[atari]

# For building our open AI environment
import gym 
from gym import envs

# For Testing random Agent actions
import random

# For Plotting result graphs
import matplotlib.pyplot as plt

# For Displaying Game Play
from IPython import display

# For neccesary model-building API's
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
import numpy as np

# For Mounting drive
from google.colab import drive

# For downloading Game Roms
import urllib.request

# Keras RL Dependancies
# For Building Game Agent
# Reinforcement learning agent we will be using
from rl.agents import DQNAgent
# Allows us to have a memory buffer from previously played games
from rl.memory import SequentialMemory
# GreedyQ allows us to find the optimal reward outcome
# Linear Annealed Policy provides Decay so we can converge when we reach optimum
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

# For changing our learning rate at the end of each EPOCH
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau


#Preparing for take off

# Mounting Google Drive folder for CS4287 Project 3
# For cloning git repo later on
drive.mount('./project_files')
!ls
! git clone https://github.com/openai/gym.git

# Importing the Atari game roms

# Roms can be found at this url
urllib.request.urlretrieve('http://www.atarimania.com/roms/Roms.rar','Roms.rar')

# Installing the roms
!pip install unrar
!unrar x Roms.rar
!mkdir rars
!mv HC\ ROMS.zip   rars
!mv ROMS.zip  rars
!python -m atari_py.import_roms rars

#Constants

#Saves the enviroment name to call later
env_name = 'SpaceInvaders-v0'
#Number of iterations
EPOCHS = 10
# To output progress
VERBOSE = 1
#Calls the SpaceInvaders-v0 enviroment from 
env = gym.make(env_name)
#Extracts state components from enviroment
height, width, channels = env.observation_space.shape
#Extracts actions from gym enviroment and returns number of actions that can be taken
actions = env.action_space.n
#Resets enviroment to starting point
obs = env.reset()

#Prints dimensions of observation space
print('Observation Space: ', env.observation_space)
#Prints number of possible actions
print('Action Space: ', env.action_space)

#Returns meaning of available actions from enviroment
env.unwrapped.get_action_meanings()

#Build Functions

# Renders Game image
# After each move we call show_obs() and pass in the env
def show_obs(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

#Passes in state components from earlier as well as action count
def build_model(height, width, channels, actions):
  #Initialises model as sequential
  model = Sequential()
  #Convolution layers added 32 filters of 8x8.
  #Stride of 4x4
  #Input shape of 3 with state components means 3 most recent frames are passed to help system understand movement in the enviroment
  model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
  #Convolution layers added 64 filters of 4x4.
  #Stride of 2x2
  model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
  # Stride = 1x1 (Px by Px)
  # Activatin RELU
  model.add(Convolution2D(64, (3,3), activation='relu'))
  #Flattens image to pass to dense layers
  model.add(Flatten())
  # Two fully connected layers using:
    # 512 UNITS - Activation ELU
    # 256 UNITS - Activation RELU
  model.add(Dense(512, activation='elu'))
  model.add(Dense(256, activation='relu'))
  #Number of outputs based on actions size so whichever one activates chooses the action
  model.add(Dense(actions, activation='linear'))
  return model

# This helper functions was made as an antidote to a commonly occuring error with Tensorflow
# It deletes our model
def fix_error(model):
  del model

# Preprocessed Game Screen
def pre_process(obs):
  obs_preprocessed = preprocess_frame(obs).reshape(88,80)
  plt.imshow(obs_preprocessed)
  plt.show()

def build_agent(model, actions):
  #LinearAnnealedPolicy is here for decay as the agent gets closer to optimum strategy
  #EpsGreedyQPolicy is for following best reward (Off policy)
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
  #Remembers the previous 3 steps for 1000 moves helping the system remember previous moves
  memory = SequentialMemory(limit=1000, window_length=3)
  dqn = DQNAgent(model=model, memory=memory, policy=policy,
                 #Dueling network allows the network to get a slight preview into alternative moves
                 enable_dueling_network=True, dueling_type='avg', 
                 nb_actions=actions, nb_steps_warmup=1000)
  return dqn

#Adaptive Learning rate function
This will reduce the learning rate as the EPOCHS

Our learning_rate starts at 0.1 and devided by 10

def adaptive_lr(EPOCHS):

    learning_rate = 0.1

    if EPOCHS > 8:
        learning_rate = learning_rate/10
    elif EPOCHS > 6:
        learning_rate = learning_rate/10
    elif EPOCHS > 4:
        learning_rate = learning_rate/10
    elif EPOCHS > 2:
        learning_rate = learning_rate/10
        
    print('Learning rate adapted: ', learning_rate)

    return learning_rate

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(adaptive_lr)
#This reduces the learning rate on plataeu
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
CALLBACKS = [lr_reducer, lr_scheduler]

#Testing

The below code will build a simple agent that take random that performs arbitrary actions, the reason for this is to compare the affects of randomly choosing actions vs applying reinforcement learning.

# Loops through each episode
for episode in range(EPOCHS):
    # Reset environment state
    obs = env.reset()
    # Flag for stopping the game IF DONE
    done = False
    score = 0 
    
    # While game is NOT finished...Keep playing
    while not done:
        # Showing our rendering method described above to display the gameplay footage
        show_obs(env)
        # Randomly choose the agents action to preform
        action = random.choice([0,1,2,3,4,5])
        # Extracting state, reward, statu, & info after applying random action to our environment
        n_state, reward, done, info = env.step(action)
        # Taking extracted score and appending it to our defind score above 
        score+=reward
        # Printing Game details
        print('Ep: ', episode, ' Scoring... ', score)

# Closing environment
env.close

#Building Our Model

# Using Adam Optimiser wit Adaptve learning rate
OPTIMISER = Adam(learning_rate=adaptive_lr(0))
# Building our model with extracted params
model = build_model(height, width, channels, actions)
# Compiling model using Mean Square Error Loss & Adam Optimiser with adaptive lr
model.compile(loss='mse', optimizer=OPTIMISER)
# Printing Model Architecture 
model.summary()

# Buildinng Our Agent
dqn = build_agent(model, actions)
# Compiling Agent using Adam optimiser with adaptive lr
dqn.compile(OPTIMISER)
# Fitting Agent, Steps set to 10000 for Validation
# Visualsise set to false as true is not compatible with Colab
# Verbose set to 1 to show progress bar
dqn.fit(env, nb_steps=10000, visualize=False, verbose=VERBOSE)

# Extracting game details using Keras rl.test()
scores = dqn.test(env, nb_episodes=15, visualize=False)
# Printing the mean of scores episode rewards
print(np.mean(scores.history['episode_reward']))

#Plots
#Returns a dict object of the episode rewards for charting
scores.history['episode_reward']

#Converting dict into more useful arrays
rewardData = list(scores.history['episode_reward'])
stepsData = list(scores.history['nb_steps'])
reward_array = np.array(rewardData)
steps_array = np.array(stepsData)

#Plotting arrays to show value of steps and rewards
plt.title("Graph of reward vs episodes")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.plot(steps_array, color="red", label="Steps")
plt.plot(reward_array, color ="green", label="Reward")
plt.legend()
plt.show()

# Replay
Here we save our agent's weights so that we can reuse them again

# Saving weights to attached drive
dqn.save_weights("./project_files/weights.h5f")
# Deleting model and agent to emulate running a fresh note book
fix_error(model)
fix_error(dqn)

# Re-building model
model = build_model(height, width, channels, actions)

# Re-building model
dqn = build_agent(model, actions)
dqn.compile(OPTIMISER)

# Loading previously saved pre-trained wieghts
dqn.load_wieghts('./project_files/weights.h5f')

# Test
scores = dqn.test(env, nb_episodes=15, visualize=False)
print(np.mean(scores.history['episode_reward']))