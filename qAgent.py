import numpy as np
import random
from collections import defaultdict

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import DeepQNetwork

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
    def __init__(self, piece):
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

#https://youtu.be/wc-FxNENg9U
class DeepComputerPlayer(Player):
    def __init__(self, piece, input_dims):
        #we will represent the input as a hashed board state so right now it'll be [1]. 
        #If we decide to move to an unhashed representation, we can try [6,7] as board state.
        self.player_type = "computer"
        self.piece = piece
        self.num_episodes = 10000
        self.max_steps_per_episode = 100

        self.learning_rate = 0.001  # alpha
        self.discount_rate = 0.99  # gamma

        self.rewards_all_episodes = []
        
        self.epsilon = 1.0
        max_mem_size=100000
        self.eps_end=0.01
        self.eps_dec=5e-4
        
        
        self.action_space = [i for i in range(NUM_ACTIONS)]
        self.mem_size = max_mem_size
        self.batch_size = 64
        self.mem_cntr = 0
        
        self.Q_eval = DeepQNetwork(self.learning_rate, n_actions=NUM_ACTIONS, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
        self.old_state = -1
        self.current_action = -1

    def chooseAction(self, state):
        
        if np.random.random() > self.epsilon:
            state_T = T.tensor([state]).float().to(self.Q_eval.device)
            #state_T = T.reshape(state_T, (1, 6, 7)) #state_T: 1 x 6 x 7
            actions = self.Q_eval.forward(state_T) 
            action = T.argmax(actions).item() #dereference tensor and get max reward action
        
        else: #exploration
            action = np.random.choice(self.action_space)
        
        self.old_state = state #update old state to current state
        self.current_action = action #sets and saves action being taken for a given move.
        
        return action

    #disables an invalid action for a certain state and reprompts the selection of another action.
    def disableaction(self, state, action, deep_invalid_actions):

        state_T = T.tensor([state]).float().to(self.Q_eval.device)
        actions = self.Q_eval.forward(state_T)
        #mask out invalid actions; normally index by batch here, but since its a single state smaple, index 0!
        actions[0][deep_invalid_actions.astype(bool)] = -999 
        action = T.argmax(actions).item() #dereference tensor and get max reward action
        
        self.old_state = state  # update old state to current state
        self.current_action = action  # sets and saves action being taken for a given move.

        return action
        
    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_cntr % self.mem_size #first empty memory slot, wraps around and overwrites after exceeding max mem size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1

    #called after droppiece so state should be new state now.
    def updateQ(self, reward, new_state):
        #don't learn until we hit a batch size
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad() #pytorch specific, zeros gradient on optimizer
        #wraparound
        max_mem = min(self.mem_cntr, self.mem_size)
        #selecting [batch_size] memory examples to create a batch; indeces for our batch 
        batch = np.random.choice(max_mem, self.batch_size, replace=False) 
        #sorted the batch indeces
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        #batch is just an index so these get the actual state values for that index
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        
        #values for the actions we actually took for each set of our memory batch
        #The indexing here describes getting the action we actually took out of the [1x6] returned from forward
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] #forward returns [batch x action x 1]?
        
        #estimate of next state
        #Todo: target network could go here
        #new_state_batch = T.reshape(new_state_batch, (self.batch_size, 6, 7)) #batch_T: 64 x 6 x 7
        q_next = self.Q_eval.forward(new_state_batch) #no dereferencing
        q_next[terminal_batch] = 0.0
        
        #belman equation
        q_target = reward_batch + self.discount_rate * T.max(q_next, dim=1)[0] #[value, index] tuple returned from T.max
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward() #backprop
        self.Q_eval.optimizer.step()
        
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end \
            else self.eps_end
            
        return
    
    def saveModel(self):
        # Specify a path
        PATH = "state_dict_model.pt"
        # Save
        T.save(self.Q_eval.state_dict(), PATH)
    
    def loadModel(self, model_obj):
        self.Q_eval = model_obj
