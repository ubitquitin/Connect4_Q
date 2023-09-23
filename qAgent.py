import numpy as np
import random
from collections import defaultdict
NUM_POSSIBLE_STATES = 74088
NUM_ACTIONS = 7

class Player:
    def __init__(self, piece):
        self.piece = piece

    #function to choose which action to take
    def chooseAction(self):
        pass

    def make_move(self):
        pass

    def get_piece(self):
        return self.piece

    def set_piece(self, piece):
        self.piece = piece

class RandomPlayer(Player):
    def __init__(self, piece):
        self.piece = piece
        self.player_type = "random"

    def chooseAction(self, state):
        action = random.randint(0,6)
        return action

class HumanPlayer(Player):
    def __init__(self, piece):
        self.piece = piece
        self.player_type = "human"

    def chooseAction(self, state):
        action = int(input("Player 1 Make your Selection (0-6):"))
        return action


class ComputerPlayer(Player):
    def __init__(self, piece, initial_state):
        self.player_type = "computer"
        self.piece = piece
        self.num_episodes = 10000
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1  # alpha
        self.discount_rate = 0.99  # gamma

        self.exploration_rate = 0.4

        self.rewards_all_episodes = []

        #{state: {0:_, 1:_, 2:_, ...}}
        self.q_table = {}

        self.old_state = -1
        self.current_action = -1

    def chooseAction(self, state):
        # if unvisited, add action for this state to the q table
        if not state in list(self.q_table.keys()):
            keys = [d for d in range(7)]
            vals = list(np.zeros(len(keys)))
            self.q_table[state] = dict(zip(keys, vals))

        action_values = list(self.q_table[state].values())

        #exploration
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0,6)

        else:
            #get the max value in q_table; for nested dict index by input state and find max value in subdictionary
            #so q_table[state] should be an array of 7 values (1 for each action), get the max of these and that is our action
            action = np.argmax(action_values)

        self.old_state = state #update old state to current state
        self.current_action = action #sets and saves action being taken for a given move.

        return action

    #disables an invalid action for a certain state and reprompts the selection of another action.
    def disableaction(self, state, action):

        self.q_table[state][action] = -999

        action_values = list(self.q_table[state].values())
        
        action = np.argmax(action_values) # get the column as action.

        self.old_state = state  # update old state to current state
        self.current_action = action  # sets and saves action being taken for a given move.

        return action

    #finds a value in the q table or initializes a new state to 0.0 if not already in dict.
    def fetchQ(self, state, action):
        if self.q_table[state] is None:
            self.q_table[state] = {action: 0.0}

        return self.q_table[state][action]

    #called after droppiece so state should be new state now.
    def updateQ(self, reward, new_state):
        greedy_state = self.old_state
        greedy_action = self.current_action

        #initialize q_table dict if its the first time we visit this state
        if not new_state in list(self.q_table.keys()):
            keys = [d for d in range(7)]
            vals = list(np.zeros(len(keys)))
            self.q_table[new_state] = dict(zip(keys, vals))

        # Best Q value for new state
        Qprime_SA = max(list(self.q_table[new_state].values()))

        #Best Q value for current state
        Q_SA = self.q_table[greedy_state][greedy_action]

        self.q_table[greedy_state][greedy_action]= Q_SA + self.learning_rate * (reward + (self.discount_rate * Qprime_SA) - Q_SA)

        return

