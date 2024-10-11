import numpy as np
import tensorflow as tf
import keras
import chessLibraryHelper as clh
import chess.pgn

clhboard = clh.clhBoard()

def main():
    q_network = keras.models.load_model('./q_network.keras')
    input = np.expand_dims(clhboard.state(), axis=0)

    output = q_network(input, training=False)
    #output = q_network.predict(input)
    #print(type(output))

    #moveInt = tf.math.argmax(output)
    moveInt = np.argmax(output)
    #print(type(moveInt))
    #print(moveInt)

    move = clhboard.moveFromInteger(moveInt)
    if move:
        #print(move)
        clhboard.push(move)
        #print(clhboard.move_stack)
        game = chess.pgn.Game.from_board(clhboard)
        print(game)

main()