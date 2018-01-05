import argparse

import gym_starcraft.my_envs.simple_mas_env as sc
import gym_starcraft.utils as utils
import gym_starcraft.configures as conf


class AttackClosestAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        # keep it wandering
        action[0] = 1
        action[1] = utils.get_closest(obs)
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip', default=conf.ip)
    parser.add_argument('--port', help='server port', default=conf.port)
    args = parser.parse_args()

    env = sc.SimpleMasEnv(args.ip, args.port, frame_skip=6, speed=10)
    env.seed(123)
    agent = AttackClosestAgent(env.action_space)

    episodes = 0
    while episodes < 50:
        obs = env.reset()
        done = False
        while not done:
            # each agent pick the action at the same time
            actions = {}
            for uid, ut in env.state['units_myself'].iteritems():
                # action = agent.act(env.state['units_enemy'])
                my_obs = obs[uid]
                action = agent.act(my_obs)
                actions[uid] = action
            # print actions
            obs, reward, done, info = env.step(actions)
        episodes += 1

    env.close()
