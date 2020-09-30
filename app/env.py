import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

class State:
    def __init__(self):
        # o tabuleiro é representado por um array n * n
        # 1 representa o símbolo do jogador que joga primeiro
        # -1 é o símbolo do outro jogador
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.end = None

    # Checa se alguém venceu o jogo ou se é um empate
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # Checa as linhas
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # Checa as colunas
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # Checa as diagonais
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # Verifica se é um empate
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # O jogo ainda continua
        self.end = False
        return self.end

    # Coloca o símbolo symbol na posição (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # Printa o tabuleiro
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')