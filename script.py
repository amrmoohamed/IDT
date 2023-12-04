import numpy as np
import pandas as pd 
import configparser
import subprocess

def remove_default_value(config_general,path):
    # Save the modified configuration files
    with open(path, 'w') as configfile:
        config_general.write(configfile)
    
    # Open the configuration file and read its content
    with open(path, 'r') as file:
        lines = file.readlines()

    # Remove the first line (the [DEFAULT] line)
    lines = lines[1:]

    # Open the configuration file again, this time in write mode
    with open(path, 'w') as file:
    # Write the modified content back to the file
        file.writelines(lines)


class FakeSectionHead(object):
  def __init__(self, fp):
      self.fp = fp
      self.sechead = '[DEFAULT]\n'

  def __iter__(self):
      return self

  def __next__(self):
    if self.sechead:
        try:
            return self.sechead
        finally:
            self.sechead = None
    else:
        line = next(self.fp)
        # Check if the line contains '=' before splitting
        if '=' in line:
            return line.split('=', 1)[0].strip() + '=' + str(line.split('=', 1)[1].strip())
        else:
            return line


config_general = configparser.ConfigParser()
with open('env/config_general.py') as fp:
   config_general.read_file(FakeSectionHead(fp))

# config = configparser.ConfigParser()
# with open('config.py') as fpp:
#    config.read_file(FakeSectionHead(fpp))

# q_table = np.load("Q_tables/original_env.npy")
# df = pd.DataFrame(q_table)
# df.to_excel("q_table.xlsx")

df = pd.read_csv('/Users/amrmohamed/Downloads/IDT/IDT.csv',index_col=1)
df.drop(df.columns[0], axis=1, inplace = True)
# for column in df.columns:
#     print(df[column].values)
# print(df.head())
#print(df.head())

# Load the configuration files
# config_general = configparser.ConfigParser()
# config_general.read('env/config_general.py')

# config = configparser.ConfigParser()
# config.read('config.py')

for scenario in df.columns:

    # Modify the configuration files based on the scenario
    step_reward = int(df[scenario]['step_reward'])
    config_general['DEFAULT']['step_reward'] = str(step_reward)
    #print(step_reward)
    passenger_not_at_loc_reward = int(df[scenario]['passenger_not_at_loc_reward'])
    config_general['DEFAULT']['passenger_not_at_loc_reward'] = str(passenger_not_at_loc_reward)
    #print(passenger_not_at_loc_reward)
    wrong_drop_reward = int(df[scenario]['wrong_drop_reward'])
    config_general['DEFAULT']['wrong_drop_reward'] = str(wrong_drop_reward)
    #print(wrong_drop_reward)
    end_of_episode_reward = int(df[scenario]['end_of_episode_reward'])
    config_general['DEFAULT']['end_of_episode_reward'] = str(end_of_episode_reward)
    #print(end_of_episode_reward)
    illegal_action_reward = int(df[scenario]['illegal_action_reward'])
    config_general['DEFAULT']['illegal_action_reward'] = str(illegal_action_reward)
    #print(illegal_action_reward)
    mapp = str(df[scenario]['Grid size'])
    #print(mapp)
    config_general['DEFAULT']['mapp'] = mapp
    try:
        scenario = str(scenario.split('.')[0])
    except:
        pass
    #print(scenario)
    config_general['DEFAULT']['approach'] = scenario




    # Run the main function in train mode
    config_general['DEFAULT']['train_flag'] = 'True'

    # Run the main function in train mode
    remove_default_value(config_general,'env/config_general.py')
    print(f"Training Scenario {scenario} in map {mapp}")
    subprocess.run(['python', 'main.py'])

    # Run the main function in test mode
    config_general['DEFAULT']['train_flag'] = 'False'

    # Save the modified configuration files
    with open('env/config_general.py', 'w') as configfile:
        config_general.write(configfile)

    # Run the main function in test mode
    remove_default_value(config_general,'env/config_general.py')
    print(f"Testing Scenario {scenario} in map {mapp}")
    subprocess.run(['python', 'main.py'])