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
        move = None
        try:
            move = self.moveFromInteger(action)
        except:
            reward = -1_000
            return self.state(), reward, self.is_game_over()

        activePlayer = self.turn
        self.push(move)

        outcome = self.outcome()
        if outcome:
            if outcome.winner == activePlayer:
                reward = 1_000
        else:
            reward = self.materialValue(activePlayer)

        return self.state(), reward, self.is_game_over()
    
    
    def moveFromInteger(self, number):
        startingSquare = number // 64
        endingSquare = number % 64
        return self.find_move(startingSquare, endingSquare)
    
    def materialValue(self, color):
        value = 0
        value += 1 * len(self.pieces(chess.PAWN, color))
        value += 3 * len(self.pieces(chess.KNIGHT, color))
        value += 3.5 * len(self.pieces(chess.BISHOP, color))
        value += 5 * len(self.pieces(chess.ROOK, color))
        value += 9 * len(self.pieces(chess.QUEEN, color))
        return value