import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import qAgent as qa
from qAgent import DeepComputerPlayer
import Board as b

#get hash for initial board to initialize computer player with.
testplayer1 = qa.RandomPlayer('X')
testplayer2 = qa.RandomPlayer('O')
initboard = b.Board('.', testplayer1, testplayer2 )
INITIAL_STATE_HASH = initboard.hash_board()

#ORDER OF OPERATIONS====
#action = agent.choose_action(observation)
#observation_new, reward, done, info = env.step(action) #do action and get the reward and game state from it
#score += reward
#agent.store_transition(observation, action, reward, observation_new, done)
#agent.learn()
#observation = new_observation
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
        winner = 0
        while not self.game_over:
            if self.turn == 0:
                deep_invalid_actions = np.zeros(qa.NUM_ACTIONS)
                observation = self.board.hash_board()
                deep_observation = self.board.get_board_as_integers()

                #Choose Action
                if isinstance(self.player1, DeepComputerPlayer):
                    action = self.player1.chooseAction(deep_observation)
                else:
                    action = self.player1.chooseAction(observation)
                
                #If valid action, drop piece and update rewards/q-table
                if self.board.is_valid_location(action):
                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player1.get_piece())
                    if self.board.in_win_state(self.player1.get_piece()):
                        print("PLAYER 1 Wins!")
                        winner = 1
                        self.game_over = True
                    
                    if self.player1.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player1.piece)
                        if isinstance(self.player1, DeepComputerPlayer):
                            self.player1.store_transition(deep_observation, action, reward, self.board.get_board_as_integers(), self.game_over)
                        self.player1.updateQ(reward, self.board.hash_board()) #reward and board "hash" arent used in DCP

                #If invalid action, choose another action
                else:
                    if isinstance(self.player1, DeepComputerPlayer):
                        #for the neural net, disable action space here instead of within the q-table. (We can't turn off the neural network weights)
                        while(not self.board.is_valid_location(action)):
                            deep_invalid_actions[action] = 1 #update mask with new invalid action.
                            action = self.player1.disableaction(deep_observation, action, deep_invalid_actions)
                            
                    elif self.player1.player_type == "computer":
                        #do computer stuff.
                        while(not self.board.is_valid_location(action)):
                            #we repeat the max(q_table) functionality in disableaction so that exploration vs. exploitation won't be rerolled.
                            action = self.player1.disableaction(observation, action)

                    elif self.player1.player_type == "random":
                        # do random stuff
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            print(self.board.print_board())
                            action = self.player1.chooseAction(observation)

                    else:
                        #Turn this into a while loop...
                        #this still lets the turn go by
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            action = self.player1.chooseAction(observation)

                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player1.get_piece())
                    if self.board.in_win_state(self.player1.get_piece()):
                        print("PLAYER 1 Wins!")
                        winner = 1
                        self.game_over = True
                    if self.player1.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player1.piece)
                        if isinstance(self.player1, DeepComputerPlayer):
                            self.player1.store_transition(deep_observation, action, reward, self.board.get_board_as_integers(), self.game_over)
                        self.player1.updateQ(reward, self.board.hash_board())

            elif self.turn == 1:
                deep_invalid_actions = np.zeros(qa.NUM_ACTIONS)
                observation = self.board.hash_board()
                deep_observation = self.board.get_board_as_integers()

                #Choose Action
                if isinstance(self.player2, DeepComputerPlayer):
                    action = self.player2.chooseAction(deep_observation)
                else:
                    action = self.player2.chooseAction(observation)
                
                #If valid action, drop the piece and update rewards/Q-table if relevant.
                if self.board.is_valid_location(action):
                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())
                    
                    if self.board.in_win_state(self.player2.get_piece()):
                        print("PLAYER 2 Wins!")
                        winner = 2
                        self.game_over = True
                        
                    if self.player2.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player2.piece)
                        if isinstance(self.player2, DeepComputerPlayer):
                            self.player2.store_transition(deep_observation, action, reward, self.board.get_board_as_integers(), self.game_over)
                        self.player2.updateQ(reward, self.board.hash_board())  #reward and board "hash" arent used in DCP

                #If invalid action, choose another action
                else:
                    
                    if isinstance(self.player2, DeepComputerPlayer):
                        #for the neural net, disable action space here instead of within the q-table. (We can't turn off the neural network weights)
                        while(not self.board.is_valid_location(action)):
                            deep_invalid_actions[action] = 1 #update mask with new invalid action.
                            action = self.player2.disableaction(deep_observation, action, deep_invalid_actions)
                    
                    elif self.player2.player_type == "computer":
                        #do computer stuff.
                        while(not self.board.is_valid_location(action)):

                            #we repeat the max(q_table) functionality in disableaction so that exploration vs. exploitation won't be rerolled.
                            action = self.player2.disableaction(observation, action)

                    elif self.player2.player_type == "random":
                        # do random stuff
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            print(self.board.print_board())
                            action = self.player2.chooseAction(observation)

                    else:
                        #Turn this into a while loop...
                        while(not self.board.is_valid_location(action)):
                            print("Please Choose another value!")
                            action = self.player2.chooseAction(observation)

                    self.board.drop_piece(self.board.get_next_open_row(action), action, self.player2.get_piece())
                    
                    if self.board.in_win_state(self.player2.get_piece()):
                        print("PLAYER 2 Wins!")
                        winner = 2
                        self.game_over = True

                    if self.player2.player_type == "computer":
                        reward = self.board.getRewardFromState(self.player2.piece)
                        if isinstance(self.player2, DeepComputerPlayer):
                            self.player2.store_transition(deep_observation, action, reward, self.board.get_board_as_integers(), self.game_over)
                        self.player2.updateQ(reward, self.board.hash_board())

            #check for full board = Draw
            if self.board.is_full():
                print("DRAW!")
                self.game_over = True

            self.board.print_board()
            # print(board.hash_board())
            self.turn = (self.turn + 1) % 2

        return winner, self.player1, self.player2


#plotting arrays for player 2
player_dos_win_history = []
player_dos_win_rate = []
player_dos_eps = []

player_uno = qa.RandomPlayer('X')
#player_dos = qa.ComputerPlayer('O')
player_dos = qa.DeepComputerPlayer('O', [42])

for i in range(10000):
    bb = b.Board('.', player_uno, player_dos)
    game = Game(player_uno,player_dos, bb)
    winner, p1, p2 = game.start_game()
    if winner == 2:
        player_dos_win_history.append(1)
    else:
        player_dos_win_history.append(0)
        
    player_dos_win_rate.append(sum(player_dos_win_history)/len(player_dos_win_history))
    player_dos_eps.append(p2.epsilon)
    print(i)

player_dos.saveModel()

player_tres = qa.HumanPlayer('X')
bb = b.Board('.', player_tres, player_dos)
game = Game(player_tres, player_dos, bb)
game.start_game()

plt.plot(np.arange(len(player_dos_win_history)), player_dos_win_rate)
plt.plot(np.arange(len(player_dos_win_history)), player_dos_eps)
plt.legend(["Win Rate", "Epsilon"], loc ="lower right")
plt.xlabel('n Games')
plt.ylabel('Win Rate')
plt.show()