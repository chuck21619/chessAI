import numpy as np
import tensorflow as tf
import keras
import chessLibraryHelper as clh
import chess.pgn
import sys

clhboard = clh.clhBoard()

def main():

    if len(sys.argv) > 1:
        clhboard.set_fen(sys.argv[1])
    state = clhboard.state()

    q_network = keras.models.load_model('./q_network.keras')
    input = np.expand_dims(state, axis=0)
    output = q_network(input, training=False)
    moveInt = np.argmax(output)

    move = clhboard.moveFromInteger(moveInt)
    if move:
        clhboard.push(move)
        game = chess.pgn.Game.from_board(clhboard)
        print(game)

main()