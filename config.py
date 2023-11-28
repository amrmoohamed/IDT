import time
import gym
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
from env.original_env import original_env
from env.env import env
from env import config_general
import numpy as np
# original_env = gym.make("Taxi-v3", render_mode="human").env
# original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = env
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
test_episodes = 1000
train_flag = True
display_flag = False
search_flag = False
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.8 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.
early_stop_condition = 10
#approach to use 
#values are 'normal','one','two','three'
mapp = '5X5' #'12X10'
approach = config_general.approach
Matrix = config_general.Matrix
#illegal_pen = config_general.illegal_action_reward
q_table_DIR = f"Q_tables/map_{mapp}_{approach}.npy"
results_DIR = f"results/map_{mapp}_{approach}.xlsx"
report_DIR = f"reports/map_{mapp}_{approach}.xlsx"
pickle_name = f"reports/map_{mapp}_{approach}.pkl"