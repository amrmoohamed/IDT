if action == 0: #move south
                                if self.desc[1 + row, 2 * col + 1] != b"-":
                                    new_row = min(row + 1, max_row)
                                else:
                                    reward = config.illegal_action_reward
                            elif action == 1: #move north
                                if self.desc[1 + row, 2 * col + 1] != b"-":
                                    new_row = max(row - 1, 0)
                                else:
                                    reward = config.illegal_action_reward
                            elif action == 2: #move east
                                if self.desc[1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, max_col)
                                else:
                                    reward = config.illegal_action_reward
                                # if config.approach == "three":
                                #     reward =  config.Matrix[row][col]
                            elif action == 3: #move west
                                if self.desc[1 + row, 2 * col] == b":": 
                                    new_col = max(col - 1, 0)
                                else:
                                    reward = config.illegal_action_reward
                                # if config.approach == "three":
                                #     reward = config.Matrix[row][col]
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 4
                                    reward = config.step_reward
                                else:  # passenger not at location
                                    reward = config.passenger_not_at_loc_reward 
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = config.end_of_episode_reward
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                    reward = config.wrong_drop_reward
                                else:  # dropoff at wrong location
                                    reward = config.wrong_drop_reward
                            if reward == 'no_reward' and (new_row != row and new_col != col):
                                if (config.approach == 'test3' or config.approach == 'test4'):
                                    reward = (config.Matrix[new_row][new_col])
                                else:
                                    reward = (config.step_reward)

                            elif reward == 'no_reward' and (new_row == row and new_col == col):
                                reward = config.illegal_action_reward

                            else:
                                if (config.approach == 'test3' or config.approach == 'test4'):
                                    reward = (config.Matrix[new_row][new_col])
                                else:
                                    reward = (config.step_reward)