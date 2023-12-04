step_reward = -10
passenger_not_at_loc_reward = -100
wrong_drop_reward = -100
end_of_episode_reward = 200
illegal_action_reward = -100
train_flag = True
approach = test5
mapp = 12*10
config_name = f'config_{mapp}.py'
module = __import__(config_name.replace('.py', ''))
matrix = module.Matrix

