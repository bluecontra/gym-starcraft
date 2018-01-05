import numpy as np

from gym import spaces
from torchcraft_py import proto
import gym_starcraft.utils as utils

import gym_starcraft.envs.starcraft_env as sc

DISTANCE_FACTOR = 16

class MasterSlaveEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=10, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(MasterSlaveEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        
        # unit_id, HP, cooldown, relative_angle, relative_distance, is_enemy
        obs_low = [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        obs_high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        for uid, action in actions.items():

            if action[0] > 0:
                target = -1
                # Attack action
                # compute the enemy id based on its position
                # target = self._get_id_by_position(action[1], action[2])
                myself_x = self.state['units_myself'][uid].x
                myself_y = self.state['units_myself'][uid].y
                degree = action[1] * 180
                distance = (action[2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself_x, -myself_y)
                target = self._get_id_by_position(x2, -y2)

                if target != -1:
                    cmds.append(proto.concat_cmd(
                        proto.commands['command_unit_protected'], uid,
                        proto.unit_command_types['Attack_Unit'], target))
            else:
                # Move action
                if self.state['units_myself'][uid] is None:
                    break
                myself_x = self.state['units_myself'][uid].x
                myself_y = self.state['units_myself'][uid].y
                degree = action[1] * 180
                distance = (action[2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself_x, -myself_y)
                cmds.append(proto.concat_cmd(
                    proto.commands['command_unit_protected'], uid,
                    proto.unit_command_types['Move'], -1, x2, -y2))

        # print cmds
        return cmds

    def _make_observation(self):
        obs_dict = {}

        # for each agent in my team, it has its own observations
        for uid, ut in self.state['units_myself'].iteritems():
            obs_dict[uid] = self._make_observation_for_agent(uid, ut)

        # print obs_dict
        return obs_dict

    def _compute_reward(self):
        reward_dict = {}
        terminal_reward = 0

        # terminal reward settings
        if self._check_done() and not bool(self.state['battle_won']):
            terminal_reward = -0.2
        if self._check_done() and bool(self.state['battle_won']):
            terminal_reward = 1
            self.episode_wins += 1
        if self.episode_steps == self.max_episode_steps:
            terminal_reward = -0.2

        # independent reward for each single agent
        for uid, ut in self.state['units_myself'].iteritems():
            my_reward = 0
            ally_num_pre, enemy_num_pre = utils.get_ally_enemy_num(self.obs_pre[uid])
            ally_num, enemy_num = utils.get_ally_enemy_num(self.obs[uid])
            my_reward = (ally_num - ally_num_pre) - (enemy_num - enemy_num_pre)

            reward_dict[uid] = my_reward + terminal_reward

        return reward_dict

    # make local observations for specific unit
    def _make_observation_for_agent(self, uid, unit):
        obs = {}
        myself = unit
        # my_view = unit.groundRange
        my_view = 1E30

        # get the observation of myself
        obs[uid] = utils.get_observation(self.observation_space.shape, myself, myself)

        # get the observation of my teammates
        for teammate_id, teammate in self.state['units_myself'].iteritems():
            if teammate_id != uid:
                temp = utils.get_observation(self.observation_space.shape, myself, teammate)
                # if the unit is in my view, add it to my observation
                if temp[4] < my_view:
                    obs[teammate_id] = temp

        # get the observation of my enemys
        for enemy_id, enemy in self.state['units_enemy'].iteritems():
            temp = utils.get_observation(self.observation_space.shape, myself, enemy)
            temp[5] = 1.0
            # if the unit is in my view, add it to my observation
            if temp[4] < my_view:
                obs[enemy_id] = temp

        return obs

    def _get_id_by_position(self, x, y):
        tid = -1
        bias_range = 1


        for uid, ut in self.state['units_enemy'].iteritems():
            enemy_x = ut.x
            enemy_y = ut.y
            dis = utils.get_distance(x, y, enemy_x, enemy_y)
            if dis < bias_range:
                bias_range = dis
                tid = uid

        return tid