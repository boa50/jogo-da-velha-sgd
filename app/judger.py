from env import State

class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None

    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, train=False, print_state=False):
        alternator = self.alternate()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            current_state, is_end = player.act()
            
            if print_state:
                current_state.print_state()
            if is_end:
                if train:
                    player = next(alternator)
                    player.backup(current_state, True)
                return current_state.winner

            self.p1.set_state(current_state)
            self.p2.set_state(current_state)