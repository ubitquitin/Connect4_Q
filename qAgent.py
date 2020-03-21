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

        self.exploration_rate = 0.2

        self.rewards_all_episodes = []
        self.q_table = defaultdict(lambda: defaultdict(int))
        #initialize all actions for current state to have a reward of 0
        self.q_table[initial_state] = [{d: 0.0} for d in range(7)]

        self.old_state = -1
        self.current_action = -1

    def chooseAction(self, state):
        # if unvisited, add action for this state to the q table
        #Todo: potentially buggy
        if self.q_table[state][0] == 0:

            self.q_table[state] = [{d: 0.0} for d in range(7)]

        values = [d for d in self.q_table[state]]
        action_values = []
        for i in values:
            b = max(i.values())
            action_values.append(b)

        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0,6)
            #if unvisited, adds the action for this state to the q_table

        else:
            #get the max value in q_table; for nested dict index by input state and find max value in subdictionary
            #so q_table[state] should be an array of 7 values (1 for each action), get the max of these and that is our action

            #print([type(d) for d in self.q_table[state]])
            max_ind = max(action_values)
            action = action_values.index(max(action_values)) #get the column as action.

        self.old_state = state #update old state to current state
        self.current_action = action #sets and saves action being taken for a given move.



        return action

    #disables an invalid action for a certain state and reprompts the selection of another action.
    def disableaction(self, state, action):
        self.q_table[state][action][action] = -1
        print(self.q_table[state][action])

        #choose a new action (max)...
        values = [d for d in self.q_table[state]]
        action_values = []
        for i in values:
            b = max(i.values())
            action_values.append(b)

        max_ind = max(action_values)
        action = action_values.index(max(action_values))  # get the column as action.

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
        state = self.old_state
        action = self.current_action

        #initialize q_table dict if its the first time we visit this state
        print(len(self.q_table[new_state].values()))
        if len(self.q_table[new_state].values()) == 0:
            self.q_table[new_state] = [{d: 0.0} for d in range(7)]

        # Update Q table
        #print(self.q_table[new_state])
        #print(self.q_table[state])
        #print(self.q_table[state][action])
        new_state_vals = []
        for i in self.q_table[new_state]:
            new_state_vals.append(list(i.values()))
        new_state_vals_flat = [item for sublist in new_state_vals for item in sublist]
        #print(new_state_vals_flat)
        #print(np.max(new_state_vals_flat))

        #Todo: picking 0 everytime
        #always will be 1 element
        state_oldaction_val = list(self.q_table[state][action])[0]
        #print((list(self.q_table[state][action]))[0])
        #print(state_oldaction_val)
        self.q_table[state][action][action] = state_oldaction_val + self.learning_rate * \
                                      (reward + (self.discount_rate * np.max(new_state_vals_flat)) - state_oldaction_val)

        return

