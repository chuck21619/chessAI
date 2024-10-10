import chess
import numpy as np
import chessLibraryHelper as clh

board = chess.Board()
board.set_fen("7k/8/5K2/6Q1/8/8/8/8 w - -")
clhboard = clh.clhBoard()
#clhboard.set_fen("7k/8/5K2/6Q1/8/8/8/8 w - -")

def main():
    #print(clhboard.fen())
    print(clhboard.state())
    #stringToNumpy()
    #print(materialValue(chess.BLACK))

def stringToNumpy():
    sample = "123123"
    numpyVersion = np.int64(sample)
    print(f"type: {type(numpyVersion)}")
    print(f"value: {numpyVersion}")

def materialValue(color):
    value = 0

    value += 1 * len(board.pieces(chess.PAWN, color))
    value += 3 * len(board.pieces(chess.KNIGHT, color))
    value += 3.5 * len(board.pieces(chess.BISHOP, color))
    value += 5 * len(board.pieces(chess.ROOK, color))
    value += 9 * len(board.pieces(chess.QUEEN, color))

    return value
        

def makeCheckmateMove():
    for move in board.legal_moves:
        if move.from_square == 38:
            if move.to_square == 54:
                print(move)
                board.push(move)

    result = board.outcome()
    print(result)


main()