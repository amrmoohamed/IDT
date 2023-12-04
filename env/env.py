from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

import gym
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled

from env import config_5X5 as env_config
from env import config_10X12 as env_config_big
import sys
from env import config_general as config


class env(Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.mapp = config.mapp
        if self.mapp == '5X5':
            self.env_config = env_config
        elif self.mapp == '12X10':
            self.env_config = env_config_big
        
        self.desc = np.asarray(self.env_config.MAP, dtype="c")
        self.locs = locs = self.env_config.locs
        self.locs_colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        self.num_states = self.env_config.num_states
        self.num_rows = self.env_config.num_rows
        self.num_columns = self.env_config.num_columns
        #self.config.mapp[:2] [-2:]
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = 6
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx: #Passenger not in the taxi and not arrived
                            self.initial_state_distrib[state] += 1
                        for action in range(self.num_actions):
                            # if config.approach == "Baseline":
                            #     reward = (
                            #         config.step_reward
                            #     )  # default reward when there is no pickup/dropoff
                            # if config.approach == "three":
                            #     reward = (
                            #         config.Matrix[new_row][new_col]
                            #     )  # default reward when there is no pickup/dropoff
                            terminated = False
                            #print(f"row is {row}, col is {col}, pass idx is {pass_idx}, dest idx is {dest_idx}, action is {action}")
                            terminated, reward, new_row, new_col, new_pass_idx = self.calculate_reward(terminated, action, row, col, pass_idx, dest_idx)
                            #print(f"reward is {reward}, new row is {new_row}, new col is {new_col}, new pass idx is {new_pass_idx}")
                            #print(reward, new_row, new_col, new_pass_idx,action)
                            # if new_row == row and new_col == col:
                            #     reward = config.no_move_reward
                            #action_mask = self.action_mask(state)
                            # if action_mask[action]==0:
                            #     reward = config.illegal_action_reward

                            #action_mask = self.action_mask(state)
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )
                            # print(self.P)
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        self.render_mode = render_mode

        # pygame utils
        self.window = None
        self.clock = None
        self.cell_size = (
            env_config.WINDOW_SIZE[0] / self.desc.shape[1],
            env_config.WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None
    


    def calculate_reward(self, terminated, action, row, col, pass_idx, dest_idx):
        # defaults
        new_row, new_col, new_pass_idx = row, col, pass_idx
        taxi_loc = (row, col)
        #reward = 'no_reward'
        if (config.approach == 'test2' or config.approach == 'test4' or config.approach == 'test6'):
            reward = (config.Matrix[new_row][new_col])
        else:
            reward = (config.step_reward)

        
        # if config.illegal_action_reward == 0:
        #     config.illegal_action_reward = config.step_reward

        if action == 0: # move south
            #if self.desc[1 + row, 2 * col + 1] != b"-":
            new_row = min(row + 1, self.max_row)
            if new_row == row:
                reward = config.illegal_action_reward
        elif action == 1: # move north
            #if self.desc[1 + row, 2 * col + 1] != b"-":
            new_row = max(row - 1, 0)
            if new_row == row:
                reward = config.illegal_action_reward
        elif action == 2: # move east
            if self.desc[1 + row, 2 * col + 2] == b":":
                new_col = min(col + 1, self.max_col)
            else:
                reward = config.illegal_action_reward
        elif action == 3: # move west
            if self.desc[1 + row, 2 * col] == b":":
                new_col = max(col - 1, 0)
            else:
                reward = config.illegal_action_reward
        elif action == 4: # pickup
            if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                new_pass_idx = 4
                #reward = config.step_reward
            else: # passenger not at location
                reward = config.passenger_not_at_loc_reward
        elif action == 5: # dropoff
            locs_without_destination = [loc for loc in self.locs if self.locs.index(loc) != dest_idx]
            #print(locs_without_destination)
            if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4: # Taxi is at destination and Passenger is inside it
                new_pass_idx = dest_idx
                terminated = True
                reward = config.end_of_episode_reward
            elif (taxi_loc in locs_without_destination)  and pass_idx == 4: # Taxi is at one of the four locs (not destination) and passenger is inside it 
                new_pass_idx = self.locs.index(taxi_loc)
                #reward = config.wrong_drop_reward  
            else: # dropoff at wrong location # won't update passenger index , as the taxi cannot drop off the passenger outside the specfied locations
                reward = config.wrong_drop_reward
        
        #if reward == 'no_reward' and (new_row != row or new_col != col):
        # elif reward == 'no_reward' and (new_row == row and new_col == col):
        #     reward = config.illegal_action_reward

        return terminated, reward, new_row, new_col, new_pass_idx


    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        i = taxi_row
        i *= env_config.row_encoding
        i += taxi_col
        i *= env_config.col_encoding
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % env_config.col_encoding)
        i = i // env_config.col_encoding
        out.append(i % env_config.row_encoding)
        i = i // env_config.row_encoding
        out.append(i)
        assert 0 <= i < env_config.decode_assertion
        return reversed(out)

    def action_mask(self, state: int):
        """Computes an action mask for the action space using the state information."""
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        if taxi_row < env_config.row_encoding-1:
            mask[0] = 1
        if taxi_row > 0:
            mask[1] = 1
        if taxi_col < env_config.row_encoding-1 and self.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            mask[2] = 1
        if taxi_col > 0 and self.desc[taxi_row + 1, 2 * taxi_col] == b":":
            mask[3] = 1
        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask
    def step(self, a):
        transitions = self.P[self.s][a]
        # print("transitions",transitions)
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        elif self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi")
            if mode == "human":
                self.window = pygame.display.set_mode(env_config.WINDOW_SIZE)
            elif mode == "rgb_array":
                self.window = pygame.Surface(env_config.WINDOW_SIZE)

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "img/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "img/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "img/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "img/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), self.cell_size)
                for file_name in file_names
            ]
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "img/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])
                self.window.blit(self.background_img, cell)
                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1] + 10))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        if pass_idx < 4:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction
        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:  # change blit order for overlapping appearance
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        out = [[c.decode("utf-8") for c in line] for line in desc]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

