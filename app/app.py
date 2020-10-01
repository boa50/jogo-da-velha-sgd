#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

from player import Player, HumanPlayer
from judger import Judger

def train(epochs, print_every_n=500):
    file = open('app/saves/metrics_all.csv', "w")
    with file:
        writer = csv.writer(file)
        writer.writerow (['win_rate1', 'win_rate2', 'draw_rate'])
    file = open('app/saves/metrics_first.csv', "w")
    with file:
        writer = csv.writer(file)
        writer.writerow (['td_error'])
    file = open('app/saves/metrics_second.csv', "w")
    with file:
        writer = csv.writer(file)
        writer.writerow (['td_error'])

    epsilon = 1
    epsilon_decay = 0.999
    epsilon_min = 0.01

    player1 = Player(epsilon=epsilon, symbol=1)
    player2 = Player(epsilon=epsilon, symbol=-1)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(train=True, print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1

        win_rate1 = player1_win / i
        win_rate2 = player2_win / i
        draw_rate = (i - (player1_win + player2_win)) / i

        metrics_file = open('app/saves/metrics_all.csv', "a")
        with metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow ([win_rate1, win_rate2, draw_rate])

        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f, draw rate: %.02f' % (i, win_rate1, win_rate2, draw_rate))

            player1.save_policy(i)
            player2.save_policy(i)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        player1.set_epsilon(epsilon)
        player2.set_epsilon(epsilon)

def compete_random(turns, policy_number):
    player1 = Player(epsilon=1, symbol=1)
    compete(player1, turns, policy_number)

def compete_greedy(turns, policy_number):
    player1 = Player(epsilon=0, symbol=1)
    player1.load_policy(policy_number)
    compete(player1, turns, policy_number)

def compete(player1, turns, policy_number):
    player2 = Player(epsilon=0, symbol=-1)
    player2.load_policy(policy_number)
    judger = Judger(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1

    draw_rate = (turns - (player1_win + player2_win)) / turns

    print('%d turns, player 1 winrate: %.02f, player 2 winrate: %.02f, draw rate: %.02f' % (turns, player1_win / turns, player2_win / turns, draw_rate))

def play(policy_number):
    player1 = HumanPlayer()
    player2 = Player(epsilon=0, symbol=-1)
    player2.load_policy(policy_number)
    while True:
        judger = Judger(player1, player2)
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")

def plot_stats(stats, smoothing_window=1, xlabel="", ylabel="", title="", legends=[]):
    fig = plt.figure(figsize=(10,5))
    has_legend = len(legends) > 0

    if smoothing_window > 1:
        if has_legend:
            for s, l in zip(stats, legends):
                stats_smoothed = pd.Series(s).rolling(smoothing_window, min_periods=smoothing_window).mean()
                plt.plot(stats_smoothed, label=l)
        else:
            for s in stats:
                stats_smoothed = pd.Series(s).rolling(smoothing_window, min_periods=smoothing_window).mean()
                plt.plot(stats_smoothed)
    else:
        if has_legend:
            for s, l in zip(stats, legends):
                plt.plot(s, label=l)
        else:
            for s, l in zip(stats, legends):
                plt.plot(s)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if has_legend:
        plt.legend()

    plt.show()

if __name__ == '__main__':
    train(int(1e5), print_every_n=1000)
    # compete_greedy(int(1e3), policy_number=100000)
    # compete_random(int(1e3), policy_number=100000)
    # play(50000)

    # df = pd.read_csv('app/saves/metrics_second.csv')
    # plot_stats([df['td_error']], smoothing_window=1000, 
    #             xlabel="Amostras", 
    #             ylabel="TD error", 
    #             title="TD error ao longo do tempo (média móvel de 1000)")

    # df = pd.read_csv('app/saves/metrics_all.csv')
    # plot_stats([df['win_rate1'], df['win_rate2'], df['draw_rate']],
    #             xlabel="Época", 
    #             ylabel="Taxa", 
    #             title="Situação por época de treinamento",
    #             legends=['Player 1 vitória', 'Player 2 vitória', 'Empate'])
