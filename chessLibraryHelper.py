import chess
import numpy as np

class clhBoard(chess.Board):
    def state(self):
        #https://www.reddit.com/r/chess/comments/11s72he/fen_to_the_matrix_data_preprocessing_for_neural/
        state = np.zeros(774)
        for i in range(len(chess.COLORS)):
            color = chess.COLORS[i]
            for k in range(len(chess.PIECE_TYPES)):
                piece = chess.PIECE_TYPES[k]
                for j in list(self.pieces(piece, color)):
                    state[(i+1)*(k+1)+j] = 1

        state[768] = self.turn
        state[769] = self.has_kingside_castling_rights(chess.WHITE)
        state[770] = self.has_kingside_castling_rights(chess.WHITE)
        state[771] = self.has_kingside_castling_rights(chess.BLACK)
        state[772] = self.has_kingside_castling_rights(chess.BLACK)

        #and target en pessant squares
        state[773] = self.halfmove_clock
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
        return self.state(), 1_000, self.is_game_over()

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