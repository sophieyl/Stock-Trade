import gym
from gym import spaces
from gym.utils import seeding

import quandl
import logging
import numpy as np
import pandas as pd

log = logging.getLogger()
log.info('%s logger started.',__name__)


class QuandlEnvSrc(object):
    '''
    Quandl-based implementation of a TradingEnv's data source.

    Pulls data from Quandl, preps for use by TradingEnv and then
    acts as data provider for each new episode.
    '''

    MinPercentileDays = 100
    QuandlAuthToken = ""  # not necessary, but can be used if desired
    Name = "WIKI/AAPL"

    # Name = "TSE/9994" # https://www.quandl.com/search (use 'Free' filter)

    def __init__(self, days=252, name=Name, auth=QuandlAuthToken, scale=True):
        self.name = name
        self.auth = auth
        self.days = days + 1
        log.info('getting data for %s from quandl...', QuandlEnvSrc.Name)
        df = quandl.get(self.name) if self.auth == '' else quandl.get(self.name, authtoken=self.auth)
        log.info('got data for %s from quandl...', QuandlEnvSrc.Name)

        df = df[~np.isnan(df.Volume)][['Open','High','Low','Close','Volume']]
        df['Return_1'] = (df.Close-df.Close.shift())/df.Close.shift()
        df['Return_3'] = (df.Close - df.Close.shift(3)) / df.Close.shift(3)
        df['Return_5'] = (df.Close - df.Close.shift(5)) / df.Close.shift(5)
        if scale:
            mean_values = df.Close.mean(axis=0)
            std_values = df.Close.std(axis=0)
            df.Close = (df.Close - np.array(mean_values)) / np.array(std_values)

        # df['buy_thres'] = df['Open'].add(df['High'])/2
        # df['sell_thres'] = df['Open'].add(df['Low']) / 2

        # self.data = df[['Close','Return','buy_thres','sell_thres']]
        df = df.dropna(axis=0,how='any')
        df['buy_price'] = 0
        df['nav'] = 1

        self.data = df[['buy_price','Close','nav','Return_1','Return_3','Return_5']]
        self.step = 0

    def reset(self):
        # we want contiguous data
        self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days) # find the date where we start
        self.step = 0

    def _step(self):
        obs = self.data.iloc[self.idx].as_matrix()
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done


class TradingSim(object):
    def __init__(self,steps, trading_cost_bps=1e-3):
        self.trading_cost_bps = trading_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.flag = 0
        self.first_buy = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.costs = np.zeros(self.steps)
        self.strategy_retrns = np.ones(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.flag = 0
        self.first_buy =0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.costs.fill(0)
        self.strategy_retrns.fill(0)
        self.mkt_retrns.fill(0)


    def _step(self,action,observation,done):
        # 'buy_price', 'Close', 'nav' ,'Return_1', 'Return_3', 'Return_5'
        # only record first buy, action = 0, buy_price = close
        # example, action: 2,1,0,0,1,2,2
        #            flag: 0,0,0,1,1,1,0
        if self.flag == 0:
            observation[0] = observation[1]
            if action == 0:
                self.costs[self.step] = self.trading_cost_bps
                self.first_buy = observation[1]
                observation[0] = self.first_buy
                self.flag += 1

        benefit = 0
        if self.flag != 0:
            observation[0] = self.first_buy
            if action == 2:
                self.costs[self.step] = self.trading_cost_bps
                benefit = observation[1] - self.first_buy
                self.flag = 0

        reward = benefit - self.costs[self.step]
        self.strategy_retrns[self.step] = reward
        self.actions[self.step] = action


        bod_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        mkt_nav = 1.0 if self.step == 0 else self.mkt_nav[self.step - 1]

        self.mkt_retrns[self.step] = (observation[1]-observation[0])/observation[0]

        if self.step == 0:
            self.mkr_first_buy = observation[1]
        if done:
            self.mkr_end = observation[1]

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strategy_retrns[self.step - 1])
            self.mkt_retrns[self.step] = observation[2]  # cost #########################
            self.mkt_nav[self.step] = mkt_nav * (1 + self.mkt_retrns[self.step - 1])


        print('reward_reality: %f, costs: %f' %(self.strategy_retrns[self.step],self.costs[self.step]))
        info = {'reward':self.strategy_retrns[self.step],'nav':self.navs[self.step]}
        self.step += 1
        return observation, reward, info



    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', ]
        # pdb.set_trace()
        df = pd.DataFrame({'action': self.actions,  # today's action (from agent)
                           'bod_nav': self.navs,  # BOD Net Asset Value (NAV)
                           'mkt_nav': self.mkt_nav,
                           'mkt_return': self.mkt_retrns,
                           'strat_return': self.strategy_retrns,
                           'costs': self.costs, },
                          columns=cols)
        df.to_csv('../../data.csv')
        return df



class TradeEnv(gym.Env):
    def __init__(self):
        self.days = 252
        self.src = QuandlEnvSrc(days=self.days)
        self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-3)
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box( self.src.min_values,
        #                                     self.src.max_values)
        self.reset()

    # get the first observation
    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # recieve action, factor,which determine and agent state, return (reward, observations)
    # action 0:buy; 1:stay 2:sell;

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        observation, done = self.src._step()
        # Close Return buy_thres  sell_thres
        # done = True, no observation #################################
        observation_, reward, info = self.sim._step(action, observation,done)
        return observation_, reward, done, info





