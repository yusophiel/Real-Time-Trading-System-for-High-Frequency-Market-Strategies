# from maze_env import Maze
import pickle
from stock_env import stock
from RL_brain import PPO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(41)

def BackTest(env, model, show_log=True, my_trick=False):
    model.eval()
    observation = env.reset(random= False)
    # print(observation.shape)
    rewards = 0
    h_out = (torch.zeros([1, 1, 64], dtype=torch.float).to(device), torch.zeros([1, 1, 64], dtype=torch.float).to(device))
    step = 0
    while True:
        h_in = h_out
        # prob, h_out, _ = model.pi(torch.from_numpy(observation).float(), h_in)
        prob, h_out, _ = model.pi(torch.from_numpy(observation).float().to(device), h_in)
        prob = prob.view(-1)
        action = torch.argmax(prob).item()
        observation_, reward, done, _ = env.step(action, random= False, steps=step, show_log=False)
        rewards = rewards + reward
        observation = observation_
        # break while loop when end of this episode
        if done:
            break
        step += 1
    print('Test total_profit:%.3f' % (env.total_profit))
    model.train()
    return env, rewards


if __name__ == "__main__":
    max_round = 55001
    # max_round = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = 'market_data_with_sentiments.csv'
    df = pd.read_csv(file_path)
    df = df.reset_index(drop=True)
    df["Real_close"] = df["close"]

    df_train = df.iloc[:-2000]
    df_test = df.iloc[-2000:]
    # df_train = df.iloc[0:2000]
    # df_test = df.iloc[2000:2500]

    scaler = StandardScaler()
    # columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', "Regulatory Impact_news",
    #                         "Technological Impact_news",
    #                         "Market Adoption Impact_news", "Macroeconomic Implications_news", "Overall Sentiment_news",
    #                         "Virality potential_x", "Informative value_x", "Sentiment polarity_x", "Impact duration_x",
    #                         "Regulatory Impact_x",
    #                         "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
    #                         "Overall Sentiment_x"]
    #
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume','EMA12','EMA26',
                            'DIFF',	'DEA',	'MACD',	'kdj_k','kdj_d','kdj_j',
                            'CCI','RSI1','RSI2','RSI3','MA5','MA10','SMA10', 'SMA50', 'DIF','DIFMA',
                            'PDI', 'MDI', 'ADX', 'ADXR', 'VOL', 'VOL5','VOL135', "Regulatory Impact_news",
                            "Technological Impact_news",
                            "Market Adoption Impact_news", "Macroeconomic Implications_news", "Overall Sentiment_news",
                            "Virality potential_x", "Informative value_x", "Sentiment polarity_x", "Impact duration_x",
                            "Regulatory Impact_x",
                            "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
                            "Overall Sentiment_x"]

    # columns_to_normalize = ['open', 'high', 'low', 'close', 'volume','EMA12','EMA26',
    #                         'DIFF',	'DEA',	'MACD',	'kdj_k','kdj_d','kdj_j',
    #                         'CCI','RSI1','RSI2','RSI3','MA5','MA10', 'DIF','DIFMA',
    #                         'PDI', 'MDI', 'ADX', 'ADXR', 'VOL', 'VOL5','VOL135', "Regulatory Impact_news",
    #                         "Technological Impact_news",
    #                         "Market Adoption Impact_news", "Macroeconomic Implications_news", "Overall Sentiment_news",
    #                         "Virality potential_x", "Informative value_x", "Sentiment polarity_x", "Impact duration_x",
    #                         "Regulatory Impact_x",
    #                         "Technological Impact_x", "Market Adoption Impact_x", "Macroeconomic Implications_x",
    #                         "Overall Sentiment_x"]

    df_train[columns_to_normalize] = scaler.fit_transform(df_train[columns_to_normalize])
    df_test[columns_to_normalize] = scaler.transform(df_test[columns_to_normalize])

    env_train = stock(df_train)

    env_test = stock(df_test)
    # pickle_file = open("trained_model/market_model.pkl", "rb")
    # model = pickle.load(pickle_file).to(device)
    # env_test, test_rewards = BackTest(env_test, model, show_log=False)
    # env_test.draw('trained_Trade/trade_Best_test-ALL.png', 'trained_Trade/profit_Best_test-ALL.png')

    # model = PPO(env_train.n_features, env_train.n_actions)
    model = PPO(env_train.n_features, env_train.n_actions).to(device)

    step = 0
    training_profit = []
    testing_profit = []

    training_reward = []
    testing_reward = []

    train_max = 0
    test_max = 0

    temperature = 2
    for episode in range(max_round):
        # initial observation
        rewards = 0
        buy_action = 0
        sell_action = 0
        not_hold_action = 0
        hold_action = 0

        h_out = (torch.zeros([1, 1, 64], dtype=torch.float).to(device), torch.zeros([1, 1, 64], dtype=torch.float).to(device))
        observation = env_train.reset(random= True)
        step = 0
        while True:
            h_in = h_out
            # prob1, h_out, logit = model.pi(torch.from_numpy(observation).float(), h_in)
            prob1, h_out, logit = model.pi(torch.from_numpy(observation).float().to(device), h_in)
            # logits = (logit / temperature)
            # prob = torch.softmax(logits, dim=2).view(-1)
            prob = prob1.view(-1)

            m = Categorical(prob)
            action = m.sample().item()
            # print(action)


            observation_, reward, done, action_state = env_train.step(action,random= True, steps=step, show_log=False)

            if action_state == 0:
                sell_action+=1

            elif action_state == 1:
                hold_action+=1
                # action = 0
            elif action_state == 2:
                not_hold_action+=1
                # action = 0
            else:
                buy_action+=1

            rewards = rewards + reward
            model.put_data((observation, action, reward, observation_, prob[action].item(), h_in, h_out, done))
            observation = observation_
            # break while loop when end of this episode
            step += 1
            # if (step % 256 == 0 or done) and episode >= 5:
            #     model.train_net()

            if done:
                break
        if episode >= 5:
            model.train_net()
            # if temperature > 1:
            #     temperature-= 0.01
        print('epoch:%d, step: %d, buy_action: %d,  sell_action: %d, hold_action: %d, not_hold_action: %d,  total_profit:%.3f' % (episode, step, buy_action, sell_action, hold_action,not_hold_action,
                                                                                                                        env_train.total_profit))

        # print('EEEENV : , buy_action: %d,  sell_action: %d, hold_action: %d, not_hold_action: %d' % (env_train.buy_action, env_train.sell_action, env_train.hold_action, env_train.not_hold_action))

        training_profit.append(env_train.total_profit)
        training_reward.append(rewards)

        if episode % 10 == 0:
            model_name = 'model_OHLCV/' + str(episode) + '.pkl'
            plt.clf()
            pickle.dump(model, open(model_name, 'wb'))

        env_test, test_rewards = BackTest(env_test,model, show_log=False)
        testing_profit.append(env_test.total_profit)
        testing_reward.append(test_rewards)
        print('Test Reward :%.3f' % test_rewards)

        if env_train.total_profit > train_max:
            train_max = env_train.total_profit
            model_name = 'model_OHLCV/train_max_model.pkl'
            plt.clf()
            pickle.dump(model, open(model_name, 'wb'))

        if env_test.total_profit > test_max:
            test_max = env_test.total_profit
            model_name = 'model_OHLCV/test_max_model.pkl'
            plt.clf()
            pickle.dump(model, open(model_name, 'wb'))
            env_test.draw('Trade/trade_Best_test-OHLCV.png', 'Trade/profit_Best_test-OHLCV.png')

        if episode % 10 == 0:
            plt.plot(training_profit)
            plt.title('PPO_Episode_train_profits')
            plt.xlabel('Episode')
            plt.ylabel('train_profits')
            plt.savefig('Reward/training_profits-OHLCV')
            plt.close()
            np.save("model_OHLCV/Train_Profits.npy", np.array(training_profit))


            plt.plot(testing_profit)
            plt.title('PPO_Episode_test_profits')
            plt.xlabel('Episode')
            plt.ylabel('test_profits')
            plt.savefig('Reward/testing_profits-OHLCV')
            plt.close()
            np.save("model_OHLCV/Test_Profits.npy", np.array(testing_profit))

            plt.plot(training_reward)
            plt.title('Train Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('Reward/Training Reward-OHLCV')
            plt.close()
            np.save("model_OHLCV/Train_Reward.npy", np.array(training_reward))

            plt.plot(testing_reward)
            plt.title('Test Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('Reward/Testing Reward-OHLCV')
            plt.close()
            np.save("model_OHLCV/Test_Reward.npy", np.array(testing_reward))

            # env_train, test_rewards = BackTest(env_train, model, show_log=False)
            # env_train.draw('Trade/trade_train-OHLCV.png', 'Trade/profit_train-OHLCV.png')

            env_test, test_rewards_2 = BackTest(env_test, model, show_log=False)
            env_test.draw('Trade/trade_test-OHLCV.png', 'Trade/profit_test-OHLCV.png')

    env_train = stock(df_train)
    env_train, test_rewards = BackTest(env_train, model, show_log=False)
    env_train.draw('Trade/trade_train.png', 'Trade/profit_train.png')

    env_test = stock(df_test)
    env_test, test_rewards_2 = BackTest(env_test, model, show_log=True)
    env_test.draw('Trade/trade_test.png', 'Trade/profit_test.png')

