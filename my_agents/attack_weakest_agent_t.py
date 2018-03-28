import argparse

import gym_starcraft.my_envs.Master_Slave_env as sc
import gym_starcraft.utils2 as utils
import gym_starcraft.configures as conf


class AttackWeakestAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        
        action[0] = 1
        t = utils.get_weakest(obs)
        # print t
        if t != -1:
            action[1] = obs[t][3]
            action[2] = obs[t][4]
        # print action
        return action


one_hot_dic = {
    0: [1,0,0,0,0,0,0,0,0,0],
    1: [0,1,0,0,0,0,0,0,0,0],
    2: [0,0,1,0,0,0,0,0,0,0],
    3: [0,0,0,1,0,0,0,0,0,0],
    4: [0,0,0,0,1,0,0,0,0,0],
    5: [0,0,0,0,0,1,0,0,0,0],
    6: [0,0,0,0,0,0,1,0,0,0],
    7: [0,0,0,0,0,0,0,1,0,0],
    8: [0,0,0,0,0,0,0,0,1,0],
    9: [0,0,0,0,0,0,0,0,0,1],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip', default=conf.ip)
    parser.add_argument('--port', help='server port', default=conf.port)
    args = parser.parse_args()

    # env = sc.SimpleMasEnv(args.ip, args.port, frame_skip=6, speed=30)
    env = sc.MasterSlaveEnv(args.ip, args.port, frame_skip=6, speed=30)
    env.seed(123)
    agent = AttackWeakestAgent(env.action_space)

    episodes = 0

    id_dic = {}
    ind = 0

    while episodes < 50:
        obs = env.reset()
        ind = 0
        for uid, ut in env.state['units_myself'].iteritems():
            id_dic[ind] = uid
            ind += 1
        # print "-"
        for uid, ut in env.state['units_enemy'].iteritems():
            id_dic[ind] = uid
            ind += 1
        print id_dic
        # print env.state['map_name']
        done = False
        while not done:
            
            
            # each agent pick the action at the same time
            actions = {}
            for uid, ut in env.state['units_myself'].iteritems():
                # print uid
                # action = agent.act(env.state['units_enemy'])
                my_obs = obs[uid]
                action = agent.act(my_obs)
                actions[uid] = action
            # print actions

            obs, reward, done, info = env.step(actions)
            
            # TO-DO
            # store transition


            # print obs
            # print reward
            # print "EP"

        # TO-DO
        # Train

        episodes += 1

    env.close()
