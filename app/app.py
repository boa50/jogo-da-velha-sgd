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

def compete(turns, policy_number):
    player1 = Player(epsilon=0, symbol=1)
    player2 = Player(epsilon=0, symbol=-1)
    judger = Judger(player1, player2)
    player1.load_policy(policy_number)
    player2.load_policy(policy_number)
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

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

if __name__ == '__main__':
    # train(int(1e5), print_every_n=1000)
    # compete(int(1e3), policy_number=10000)
    # play(10000)
