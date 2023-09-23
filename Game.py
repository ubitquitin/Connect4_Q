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
    def __init__(self, player1, player2, board):
        self.game_over = False
        self.turn = 0
        self.player1 = player1
        self.player2 = player2
        self.player1type = player1.player_type
        self.player2type = player2.player_type
        self.player1piece = player1.piece
        self.player2piece = player2.piece
        self.board = board


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
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            print(self.board.print_board())
                            action = self.player1.chooseAction(self.board.hash_board())

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
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            print(self.board.print_board())
                            action = self.player1.chooseAction(self.board.hash_board())

                    else:
                        #Turn this into a while loop...
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            action = self.player2.chooseAction(self.board.hash_board())

                        self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())

                if self.board.in_win_state(self.player2.get_piece()):
                    print("PLAYER 2 Wins!")
                    self.game_over = True

            #check for full board = Draw
            if self.board.is_full():
                print("DRAW!")
                self.game_over = True

            self.board.print_board()
            # print(board.hash_board())
            self.turn = (self.turn + 1) % 2

        return 0



player_uno = qa.RandomPlayer('X')
player_dos = qa.ComputerPlayer('O', INITIAL_STATE_HASH)


for i in range(100000):
    bb = b.Board('.', player_uno, player_dos)
    game = Game(player_uno,player_dos, bb)
    game.start_game()
    print(i)

#print(player_dos.q_table)

#Todo: Computer player is learning but not well

player_tres = qa.HumanPlayer('X')
bb = b.Board('.', player_tres, player_dos)
game = Game(player_tres, player_dos, bb)
game.start_game()