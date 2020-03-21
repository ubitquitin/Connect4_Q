import numpy as np

class Board():
    def __init__(self, fill_char, player1, player2):
        self.ROW_COUNT = 6
        self.COL_COUNT = 7
        self.fill_char = fill_char
        self.board = np.full((self.ROW_COUNT, self.COL_COUNT), self.fill_char)
        self.player1_piece = player1.get_piece()
        self.player2_piece = player2.get_piece()

    def get_board(self):
        return self.board

    def is_full(self):
        for i in range(self.COL_COUNT):
            if self.board[5][i] == self.fill_char:
                return False

        return True

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        #check if top row is free so you can drop piece in column
        #if this returns false then column is full
        if col > 6 or col < 0:
            return False
        return self.board[5][col] == self.fill_char

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            #loop through rows in chosen column and find the first 0 (searches bottom to top)
            if self.board[r][col] == self.fill_char:
                return r

    # just changes orientation of numpy matrix to connect 4 board (bottom - up)
    def print_board(self):
        print(np.flip(self.board, 0))

    def in_win_state(self, piece):
        # Check horizontal locations for win
        for c in range(self.COL_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and \
                        self.board[r][c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.COL_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and \
                        self.board[r + 3][c] == piece:
                    return True

        # Check positively sloped diagonal locations for win
        for c in range(self.COL_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][c + 2] == piece and \
                        self.board[r + 3][c + 3] == piece:
                    return True

        # Check negatively sloped diagonal locations for win
        for c in range(self.COL_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][c + 2] == piece and \
                        self.board[r - 3][c + 3] == piece:
                    return True

    #hashes board state as an integer
    def hash_board(self):
        const = []
        hash_sum = 0
        hash_sum = np.int64(hash_sum)

        #list of size 42 (1 entry for each row,col). Flattened board constants.
        for row in range(self.ROW_COUNT):
            for col in range(self.COL_COUNT):
                if(self.board[row][col] == self.fill_char):
                    const.append(0)
                elif(self.board[row][col] == self.player1_piece):
                    const.append(1)
                elif(self.board[row][col] == self.player2_piece):
                    const.append(2)
                else:
                    const.append(0)

        #hash using base 3
        counter = 41

        #make initial states have high hash changes
        const = const[::-1]

        #print(const)
        while(counter >= 0):
            hash_sum += (const[counter] * np.power(3, counter))
            counter = counter - 1


        return hash_sum

    def getRewardFromState(self, player_piece):
        if player_piece == self.player1_piece:
            if self.in_win_state(self.player1_piece):
                return 1
            elif self.in_win_state(self.player2_piece):
                return -3
            else:
                #reward for being alive
                return 0

        elif player_piece == self.player2_piece:
            if self.in_win_state(self.player2_piece):
                return 1
            elif self.in_win_state(self.player1_piece):
                return -3
            else:
                return 0

        else:
            return 0.5