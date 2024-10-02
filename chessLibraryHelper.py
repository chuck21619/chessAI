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
    
