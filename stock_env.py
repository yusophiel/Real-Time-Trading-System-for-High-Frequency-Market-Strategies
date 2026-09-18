# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class stock:

    def __init__(self, df, init_money=10000000, window_size=24):

        self.n_actions = 3
        self.n_features = (window_size) * 44 + 1

        self.trend = df['Real_close'].values
        self.close = df['close'].values
        self.volume = df['volume'].values
        self.high = df['high'].values
        self.low = df['low'].values
        self.open = df['open'].values

        self.regulatory_impact_news = df['Regulatory Impact_news'].values
        self.technological_impact_news = df['Technological Impact_news'].values
        self.market_adoption_impact_news = df['Market Adoption Impact_news'].values
        self.macroeconomic_implications_news = df['Macroeconomic Implications_news'].values
        self.overall_sentiment_news = df['Overall Sentiment_news'].values
        self.virality_potential_x = df['Virality potential_x'].values
        self.informative_value_x = df['Informative value_x'].values
        self.sentiment_polarity_x = df['Sentiment polarity_x'].values
        self.impact_duration_x = df['Impact duration_x'].values
        self.regulatory_impact_x = df['Regulatory Impact_x'].values
        self.technological_impact_x = df['Technological Impact_x'].values
        self.market_adoption_impact_x = df['Market Adoption Impact_x'].values
        self.macroeconomic_implications_x = df['Macroeconomic Implications_x'].values
        self.overall_sentiment_x = df['Overall Sentiment_x'].values


        self.df = df
        self.init_money = init_money

        self.window_size = window_size
        self.t = self.window_size
        self.buy_rate = 0.001
        self.buy_min = 5
        self.sell_rate = 0.001
        self.sell_min = 5
        self.stamp_duty = 0.001

        self.hold_money = self.init_money
        self.buy_num = 0
        self.hold_num = 0
        self.stock_value = 0
        self.market_value = 0
        self.last_value = self.init_money
        self.total_profit = 0
        self.reward = 0
        self.states_sell = []
        self.states_buy = []
        self.no_trade_duration = 0

        self.profit_rate_account = []
        self.profit_rate_stock = []

        self.buy_action = 0
        self.sell_action = 0
        self.not_hold_action = 0
        self.hold_action = 0

    def reset(self, random, init_money=10000000):
        self.init_money = init_money
        self.hold_money = self.init_money
        self.buy_num = 0
        self.hold_num = 0
        self.stock_value = 0
        self.market_value = 0
        self.last_value = self.init_money
        self.total_profit = 0
        self.reward = 0
        self.states_sell = []
        self.states_buy = []
        if random:
            self.t = np.random.randint(self.window_size,(len(self.trend) - 100))
        else:
            self.t = self.window_size

        self.profit_rate_account = []
        self.profit_rate_stock = []
        self.no_trade_duration = 0

        self.buy_action = 0
        self.sell_action = 0
        self.not_hold_action = 0
        self.hold_action = 0

        return self.get_state(self.t)


    def get_state(self, t):
        if self.hold_num > 0 :
            hold_state = 1
        else:
            hold_state = 0
        state = [hold_state]

        state.extend(
            np.array(self.df[['open', 'high', 'low', 'close', 'volume','EMA12','EMA26',
                    'DIFF',	'DEA',	'MACD',	'kdj_k','kdj_d','kdj_j',
                    'CCI','RSI1','RSI2','RSI3','MA5','MA10','SMA10', 'SMA50', 'DIF','DIFMA',
                    'PDI', 'MDI', 'ADX', 'ADXR', 'VOL', 'VOL5','VOL135',"Regulatory Impact_news",
                            "Technological Impact_news",
                            "Market Adoption Impact_news", "Macroeconomic Implications_news", "Overall Sentiment_news",
                            "Virality potential_x", "Informative value_x", "Sentiment polarity_x", "Impact duration_x",
                            "Regulatory Impact_x",
                            "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
                            "Overall Sentiment_x"]].iloc[t - 24: t]).reshape(-1))

            # np.array(self.df[['open', 'high', 'low', 'close', 'volume', 'EMA12', 'EMA26',
            #                   'DIFF', 'DEA', 'MACD', 'kdj_k', 'kdj_d', 'kdj_j',
            #                   'CCI', 'RSI1', 'RSI2', 'RSI3', 'MA5', 'MA10', 'DIF', 'DIFMA',
            #                   'PDI', 'MDI', 'ADX', 'ADXR', 'VOL', 'VOL5', 'VOL135', "Regulatory Impact_news",
            #                   "Technological Impact_news",
            #                   "Market Adoption Impact_news", "Macroeconomic Implications_news",
            #                   "Overall Sentiment_news",
            #                   "Virality potential_x", "Informative value_x", "Sentiment polarity_x",
            #                   "Impact duration_x",
            #                   "Regulatory Impact_x",
            #                   "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
            #                   "Overall Sentiment_x"]].iloc[t - 24: t]).reshape(-1))
        # state.extend(
        #     np.array(self.df[['open', 'high', 'low', 'close', 'volume', "Regulatory Impact_news",
        #                     "Technological Impact_news",
        #                     "Market Adoption Impact_news", "Macroeconomic Implications_news", "Overall Sentiment_news",
        #                     "Virality potential_x", "Informative value_x", "Sentiment polarity_x", "Impact duration_x",
        #                     "Regulatory Impact_x",
        #                     "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
        #                     "Overall Sentiment_x"]].iloc[t - 24: t]).reshape(-1))

        # print(t)
        # print(self.df['open'].iloc[t - 24: t])
        return np.array(state)

    def buy_stock(self):
        self.buy_num = self.hold_money * (1 - self.buy_rate ) / (self.trend[self.t])  # 买入手数

        tmp_money = self.trend[self.t] * self.buy_num * self.buy_rate

        self.hold_num += self.buy_num

        self.stock_value += self.trend[self.t] * self.buy_num

        self.hold_money = self.hold_money - self.trend[self.t] * self.buy_num - tmp_money

        # print(self.hold_money)
        self.states_buy.append(self.t)

    def sell_stock(self, sell_num):
        tmp_money = sell_num * self.trend[self.t]
        service_change = tmp_money * self.sell_rate
        # if service_change < self.sell_min:
        #     service_change = self.sell_min
        # stamp_duty = self.stamp_duty * tmp_money
        # self.hold_money = self.hold_money + tmp_money - service_change - stamp_duty
        self.hold_money = self.hold_money + tmp_money - service_change
        self.hold_num = 0
        self.stock_value = 0
        self.states_sell.append(self.t)

    def trick(self):
        if self.df['close'][self.t] >= self.df['ma21'][self.t]:
            return True
        else:
            return False

    def step(self, action, random, steps, show_log=False):
        # 0: Hold and sell
        # 1: Hold and do not sell
        # 2: Not hold
        if action == 1 and self.hold_money >= (self.trend[self.t]):
            buy_ = True
            action_state = 3
            self.no_trade_duration = 0
            self.buy_action += 1
            self.buy_stock()
            if show_log:
                print('day:%d, buy price:%f, buy num:%d, hold num:%d, hold money:%.3f' % \
                      (self.t, self.trend[self.t], self.buy_num, self.hold_num, self.hold_money))
        elif action == 2 and self.hold_num > 0:
            action_state = 0
            self.no_trade_duration = 0
            self.sell_action += 1

            self.sell_stock(self.hold_num)
            # print("Action: ", action)
            # print("Hold_num: ", self.hold_num)
            # print("Sell_action: ", self.sell_action)
            # print("Buy_action: ", self.buy_action)
            # print("/////////////")
            # if show_log:
            #     print(
            #         'day:%d, sell price:%f, total balance %f,'
            #         % (self.t, self.trend[self.t], self.hold_money)
            #     )
        else:
            self.no_trade_duration += 1
            if self.hold_num > 0:
                self.hold_action += 1
                action_state = 1
            else:
                self.not_hold_action += 1
                action_state = 2

            # if my_trick and self.hold_num > 0 and not self.trick():
            #     self.sell_stock(self.hold_num)
            #     if show_log:
            #         print(
            #             'day:%d, sell price:%f, total balance %f,'
            #             % (self.t, self.trend[self.t], self.hold_money)
            #         )

        self.stock_value = self.trend[self.t] * self.hold_num
        self.market_value = self.stock_value + self.hold_money
        # print(f"Stock Value: {self.stock_value}, Hold Money: {self.hold_money}, Market Value: {self.market_value}, Initial Money: {self.init_money}")
        self.total_profit = self.market_value - self.init_money

        price_change_reward = (self.trend[self.t + 1] - self.trend[self.t]) / self.trend[self.t]
        # market_value_change_reward = (self.market_value - self.last_value) / self.last_value
        # long_term_reward = self.total_profit / self.init_money

        # reward = 0.5 * price_change_reward
        reward = price_change_reward

        if action_state == 0:
            # self.reward = -reward - 0.01 + 0.5 * long_term_reward
            self.reward = -reward - 0.0001

        elif action_state == 1:
            if reward>0:
                self.reward = reward * 0.8
            else:
                self.reward = reward * 1.2
        elif action_state == 3:
            # self.reward = reward - 0.01 + 0.5 * long_term_reward
            self.reward = reward - 0.0001
        else:
            if reward > 0:
                self.reward = -reward * 0.2



        self.last_value = self.market_value

        self.profit_rate_account.append((self.market_value - self.init_money) / self.init_money)
        self.profit_rate_stock.append(
            (self.trend[self.t] - self.trend[self.window_size - 1]) / self.trend[self.window_size - 1])
        done = False

        self.t = self.t + 1

        if random:
            if steps >=2000 or (self.t == len(self.trend) - 2):
                done = True
        else:
            if self.t == len(self.trend) - 2:
                done = True

        s_ = self.get_state(self.t)
        reward = self.reward
        return s_, reward, done, action_state

    def get_info(self):
        return self.states_sell, self.states_buy, self.profit_rate_account, self.profit_rate_stock

    def draw(self, save_name1, save_name2):
        states_sell, states_buy, profit_rate_account, profit_rate_stock = self.get_info()
        invest = profit_rate_account[-1]
        total_gains = self.total_profit
        close = self.trend
        fig = plt.figure(figsize=(15, 5))
        plt.plot(close, color='r', lw=2.)
        plt.plot(close, 'v', markersize=8, color='k', label='selling signal', markevery=states_sell)
        plt.plot(close, '^', markersize=8, color='m', label='buying signal', markevery=states_buy)
        plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
        plt.legend()
        plt.savefig(save_name1)
        plt.close()

        fig = plt.figure(figsize=(15, 5))
        plt.plot(profit_rate_account, label='my account')
        plt.plot(profit_rate_stock, label='stock')
        plt.legend()
        plt.savefig(save_name2)
        plt.close()

        np.save("trained_Trade/Test_ALL_profit.npy", np.array(profit_rate_account))



