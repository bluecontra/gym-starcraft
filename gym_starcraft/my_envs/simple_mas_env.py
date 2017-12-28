import numpy as np

from gym import spaces
from torchcraft_py import proto
import gym_starcraft.utils as utils

import gym_starcraft.envs.starcraft_env as sc

DISTANCE_FACTOR = 16

class SimpleMasEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=10, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(SimpleMasEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # is enemy, health, cooldown, ground range, degree, distance, tag
        
        obs_low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        obs_high = [1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0]
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def _make_commands(self, actions):
        cmds = []
        if self.state is None or actions is None:
            return cmds

        # myself_id = None
        # myself = None
        # enemy_id = None
        # enemy = None
        # for uid, ut in self.state['units_myself'].iteritems():
        #     myself_id = uid
        #     myself = ut
        # for uid, ut in self.state['units_enemy'].iteritems():
        #     enemy_id = uid
        #     enemy = ut
        for uid, action in actions.items():

            if action[0] > 0:
                target = int(action[1])
                # Attack action
                # if self.state['units_enemy'][int(action[1])] is None or self.state['units_myself'][uid] is None:
                    # break
                # TODO: compute the enemy id based on its position
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
        myself = None
        # enemy = None

        # for each agent in my team, it has its own observations
        for uid, ut in self.state['units_myself'].iteritems():
            my_dict = {}

            myself = ut
            # get the observation of myself
            my_dict[uid] = utils.get_observation(self.observation_space.shape, myself, myself)
            # get the observation of my teammates
            for teammate_id, teammate in self.state['units_myself'].iteritems():
                if teammate_id != uid:
                    my_dict[teammate_id] = utils.get_observation(self.observation_space.shape, myself, teammate)
            # get the observation of my enemys
            for enemy_id, enemy in self.state['units_enemy'].iteritems():
                temp = utils.get_observation(self.observation_space.shape, myself, enemy)
                temp[0] = 1.0
                my_dict[enemy_id] = temp

            obs_dict[uid] = my_dict

        # print obs_dict
        return obs_dict

    def _compute_reward(self):
        reward = 0
        # if self.obs[5] + 1 > 1.5:
        #     reward = -1
        # if self.obs_pre[6] > self.obs[6]:
        #     reward = 15
        # if self.obs_pre[1] > self.obs[1]:
        #     reward = -10
        if self._check_done() and not bool(self.state['battle_won']):
            reward = -500
        if self._check_done() and bool(self.state['battle_won']):
            reward = 1000
            self.episode_wins += 1
        if self.episode_steps == self.max_episode_steps:
            reward = -500
        return reward
