import argparse

import gym_starcraft.my_envs.simple_mas_env as sc


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

class WanderingAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        action = self.action_space.sample()
        # keep it wandering
        action[0] = -1
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip', default=conf.IP)
    parser.add_argument('--port', help='server port', default="11111")
    args = parser.parse_args()

    env = sc.SimpleMasEnv(args.ip, args.port, frame_skip=6, speed=30)
    env.seed(123)
    agent = WanderingAgent(env.action_space)

    episodes = 0
    interactions = 0
    while episodes < 50:
        obs = env.reset()
        done = False
        while not done:
            # each agent pick the action at the same time
            actions = {}
            for uid, ut in env.state['units_myself'].iteritems():
                action = agent.act()
                actions[uid] = action
            obs, reward, done, info = env.step(actions)
            interactions += 1
            if interactions == 10:
                for uid, o in obs.items():
                    print str(uid)
                    print o
        episodes += 1

    env.close()
