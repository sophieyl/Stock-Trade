import numpy as np



class DQN(object):
    def __init__(
            self,
            e_greedy=0.9,):
        self.epsilon = e_greedy



    def _build_net(self):

    # observation = dict('opening':, 'highest':, 'lowest':, 'closing':)
    def choose_action(self, observation):
        observation['closing']
        self.purchasing_power =
        self.

        if np.random.uniform() < self.epsilon:
            pass
        else:
            action = np.random.randint(0,self.n_actions)


    def learn(self):
        pass