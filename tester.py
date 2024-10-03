import chess

board = chess.Board()
board.set_fen("7k/8/5K2/6Q1/8/8/8/8 w - - 0 1")
print(board.turn)

for move in board.legal_moves:
    if move.from_square == 38:
        if move.to_square == 54:
            print(move)
            board.push(move)

#board.push("Qg7#")

result = board.outcome()
print(result)
print(result.winner)