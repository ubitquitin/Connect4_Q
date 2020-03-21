import numpy as np
import pygame
import sys
import qAgent as qa
import Board as b

#get hash for initial board to initialize computer player with.
testplayer1 = qa.RandomPlayer('X')
testplayer2 = qa.RandomPlayer('O')
initboard = b.Board('.', testplayer1, testplayer2 )
INITIAL_STATE_HASH = initboard.hash_board()

class Game():
    def __init__(self, player1type, player2type, fillpiece, player1piece, player2piece):
        self.game_over = False
        self.turn = 0
        self.player1type = player1type
        self.player2type = player2type

        if player1type == 'computer':
            self.player1 = qa.ComputerPlayer(player1piece, INITIAL_STATE_HASH)
        elif player1type == 'random':
            self.player1 = qa.RandomPlayer(player1piece)
        else:
            self.player1 = qa.HumanPlayer(player1piece)

        if player2type == 'computer':
            self.player2 = qa.ComputerPlayer(player2piece, INITIAL_STATE_HASH)
        elif player2type == 'random':
            self.player2 = qa.RandomPlayer(player2piece)
        else:
            self.player2 = qa.HumanPlayer(player2piece)

        self.board = b.Board(fillpiece, self.player1, self.player2)

    def start_game(self):
        self.board.print_board()
        while not self.game_over:
            if self.turn == 0:

                action = self.player1.chooseAction(self.board.hash_board())
                if self.board.is_valid_location(action):
                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player1.get_piece())
                    if self.player1.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player1.piece)
                        self.player1.updateQ(reward, self.board.hash_board())

                else:
                    if self.player1.player_type == "computer":
                        #do computer stuff.
                        while(not self.board.is_valid_location(action)):

                            #we repeat the max(q_table) functionality in disableaction so that exploration vs. exploitation won't be rerolled.
                            action = self.player1.disableaction(self.board.hash_board(), action)



                    if self.player2.player_type == "random":
                        # do random stuff
                        print("Why would a randomint(0,6) give you an invalid number?")

                    else:
                        #Turn this into a while loop...
                        #this still lets the turn go by
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            action = self.player1.chooseAction(self.board.hash_board())

                        self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())


                if self.board.in_win_state(self.player1.get_piece()):
                    print("PLAYER 1 Wins!")
                    self.game_over = True

            elif self.turn == 1:
                action = self.player2.chooseAction(self.board.hash_board()) #getting a long array currently!!!!****
                if self.board.is_valid_location(action):
                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())
                    if self.player2.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player2.piece)
                        self.player2.updateQ(reward, self.board.hash_board())

                else:
                    if self.player2.player_type == "computer":
                        #do computer stuff.
                        while(not self.board.is_valid_location(action)):

                            #we repeat the max(q_table) functionality in disableaction so that exploration vs. exploitation won't be rerolled.
                            action = self.player2.disableaction(self.board.hash_board(), action)



                    if self.player2.player_type == "random":
                        # do random stuff
                        print("Why would a randomint(0,6) give you an invalid number?")

                    else:
                        #Turn this into a while loop...
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            action = self.player2.chooseAction(self.board.hash_board())

                        self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())

                if self.board.in_win_state(self.player2.get_piece()):
                    print("PLAYER 2 Wins!")
                    self.game_over = True

            self.board.print_board()
            # print(board.hash_board())
            self.turn = (self.turn + 1) % 2

        return 0


#game = Game('human', 'computer', '.', 'X', 'O')
game = Game('computer', 'computer', '.', 'X', 'O')
game.start_game()
#Todo: Persist Q-table across several games... this will involve not constructing and deconstructing the q_table but making it a global var of game?
#Todo: Pass player objects into game.startgame() call so the players don't get deconstructed. Replace self.player1/2 with input parameters.
print(game.player1.q_table)
print(game.player2.q_table)
#can do something like for i in range(1000): and nest everything below in that with a computer player to keep training
#the computer and keep it/'s Q table.