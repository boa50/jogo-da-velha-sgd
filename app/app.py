#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from player import Player, HumanPlayer
from judger import Judger

def train(epochs, print_every_n=500):
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
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))

            player1.save_policy(i)
            player2.save_policy(i)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        player1.update_epsilon(epsilon)
        player2.update_epsilon(epsilon)

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
    train(int(1e5), print_every_n=1000)
    # compete(int(1e3), policy_number=9)
    # play(9)
