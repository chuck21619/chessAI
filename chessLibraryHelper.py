import chess
import numpy as np

class clhBoard(chess.Board):
    def state(self):
        #https://www.reddit.com/r/chess/comments/11s72he/fen_to_the_matrix_data_preprocessing_for_neural/
        state = np.zeros(1047)
        for i in range(64):
            piece = self.piece_at(i)
            if piece != None:
                onehotencodingindex = piece.piece_type * (piece.color + 1) * i
                state[onehotencodingindex] = 1

        indexTurn = 1025
        state[indexTurn] = self.turn

        indexWhiteCastleKingSide = 1026
        indexWhiteCastleQueenSide = 1027
        indexBlackCastleKingSide = 1028
        indexBlackCastleQueenSide = 1029
        state[indexWhiteCastleKingSide] = self.has_kingside_castling_rights(chess.WHITE)
        state[indexWhiteCastleQueenSide] = self.has_kingside_castling_rights(chess.WHITE)
        state[indexBlackCastleKingSide] = self.has_kingside_castling_rights(chess.BLACK)
        state[indexBlackCastleQueenSide] = self.has_kingside_castling_rights(chess.BLACK)


        #1030-1045 target en pessant squares
        indexHalfmoveClock = 1046
        state[indexHalfmoveClock] = self.halfmove_clock
        return state

        
    
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