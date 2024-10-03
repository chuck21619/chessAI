import chess

def fenToState(fenString):
    firstSplit = fenString.split("/")
    secondSplit = firstSplit[-1].split()
    firstSplit.pop()
    firstSplit[-1] = secondSplit[0]
    return firstSplit + secondSplit


class clhBoard(chess.Board):
    def state(self):
        return fenToState(self.fen())
    
    def step(self, action):
        reward = 0

        if self.isLegalMove(action):
            reward = -10_000
        
        activePlayer = self.turn

        wonGame = False
        lostGame = False

        result = self.outcome()
        self.set_fen(action)
        #if won, reward = 10_000
        #if illegal, reward = -10_000
        #else reward = difference in material
        return self.state(), reward, self.is_game_over(), ""
    
    def isLegalMove(self, FENstring):
        allowedFENs = []

        for legal_move in self.legal_moves():
            self.push(legal_move)
            allowedFENs.append(self.fen())
            self.pop()

        self.set_fen(FENstring)
        if FENstring in allowedFENs:
            return True
        
        return False