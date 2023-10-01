'''
In Contextual Multi-Armed Bandit (C-MAB), the machine learning (ML) agent 
uses a table to store values of actions for different contexts. This
context-action value table can be replaced by a feed forward neural network 
(FFNN). There are several advantages to use a FFNN instead of a table. 
FFNN can better handle context with a huge space, and can deal with context 
containing continuous values. Besides, FFNN can natively predict action values
for unseen contexts whereas table-driven C-MAB cannot and it must use a 
separate technique such as KNN to infer the action values. However, FFNN 
requires more data to get trained.

The following is the procedure:
- For each episode:
  - a number of users are served, and this number is predefined
  - for each user service
    - the context is obtained from the environment, which is the user age
      group in our example
    - the ML agent picks an action either (i) picking randomly during exploration, 
      or (ii) picking whose value is the highest among all actions during exploitation
    - the picked action is executed
    - the corresponding reward is observed from the environment, which is 
      1 if the user clicked the offered ad, or 0 otherwise
    - the context-action-reward outcome is kept in a memory
  - at the end of an episode, the memory is used to train the FFNN
  - the memory is cleared, and the ML agent proceeds to the next episode
'''

import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

######################################################################
## User behaviour matrix for the environment (static class)
######################################################################

class Ad:

    Type = {       #  age group
      #  arm       #  <25   <35   <45    <55   >55 
      #  --------------------------------------------
        "toys"     : [0.80, 0.15, 0.10, 0.05, 0.05],
        "cars"     : [0.05, 0.50, 0.30, 0.15, 0.10],
        "sports"   : [0.15, 0.30, 0.40, 0.30, 0.30],
        "holidays" : [0.05, 0.20, 0.35, 0.50, 0.50],
        "foods"    : [0.05, 0.25, 0.25, 0.40, 0.60]
    }
    AllArms = list(Type.keys()) # list of all ad types
    ArmSize = len(AllArms)
    AgeGroupSize = len(list(Type.values())[0])
    AllAgeGroups = range(AgeGroupSize)

######################################################################
## Client profile
######################################################################

class Client:
    def __init__(self, age_group=None):
        if age_group is None:
            self.group = random.randint(0,Ad.AgeGroupSize-1)
        else:
            self.group = age_group

    def will_click(self, ad) -> bool:
        '''Will this client click this advert?'''
        click_prob = random.randint(0,99)
        if click_prob<100*Ad.Type[ad][self.group]:
            return True
        return False

######################################################################
## Machine Learning Model
######################################################################

class ContextActionTable:
    '''This is a class to keep the memory during an episode. For this particular
    example, the memory is the context-action value table.'''

    def __init__(self,num_contexts,num_actions):
        self.num_contexts = num_contexts
        self.num_actions = num_actions
        self.clear()

    def clear(self):
        '''Clear all contents in the table.'''
        self.sum   = [[0 for i in range(self.num_actions)] for j in range(self.num_contexts)]
        self.count = [[0 for i in range(self.num_actions)] for j in range(self.num_contexts)]
        self.value = [[0 for i in range(self.num_actions)] for j in range(self.num_contexts)]

    def update_reward(self,context,action,value):
        '''Update the reward into the table.'''
        self.value[context][action]  = value
        self.sum[context][action]   += value
        self.count[context][action] += 1
    
    def get_mean_value(self,context,action,default=0):
        '''Return the mean value of the (context,action), or the default value if
        there is no update for this action.'''
        if self.count[context][action]==0: return default
        return self.sum[context][action]/self.count[context][action]

    def get_last_value(self,context,action,default=0):
        '''Return the last seen value of the (context,action), or the default value if
        there is no update for this action.'''
        if self.count[context][action]==0: return default
        return self.value[context][action]


####################################################################
## main loop
####################################################################

def array_to_str(array):
    '''This function convert a one-dimensional array to a formatted string for printing.'''
    array_str = "["
    for element in array:
        array_str += f"----," if element is None else f"{element:4.2f},"
    return array_str[:-1] + "]"

def context_array(context_idx:int):
    '''Convert context into an array form for the inputs of FFNN, 
    e.g. if context=3, then array = [0,0,0,1,0].'''
    array = [0]*Ad.AgeGroupSize
    array[context_idx] = 1
    return array


if __name__ == "__main__":

    ## setup environment parameters
    num_users = 20000    # number of users to visit the website
    num_clicks = 0       # number of clicks collected
    animation  = True # True/False

    ## setup training arrangement
    episode_len = 500         # the length of an episode for each training
    exploration_round = 8000  # explore so many rounds first
    memory = []               # the memory to keep (context,action,reward)
    alpha = 0.2               # learning rate

    ## setup context-action tables
    num_contexts=Ad.AgeGroupSize
    num_actions=Ad.ArmSize
    context_action_life = ContextActionTable(num_contexts, num_actions)
    context_action_episode = ContextActionTable(num_contexts, num_actions)

    ## setup FFNN
    model = Sequential()
    model.add(Dense(units=20, activation='relu', input_dim=num_contexts))
    model.add(Dense(units=50, activation='tanh'))
    model.add(Dense(units=num_actions))
    model.compile(loss='mean_squared_error', optimizer='adam')

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    print(f"Testing Deep Reinforcement Learning\n")
    for round in range(num_users):

        ## a user has visited the website
        user = Client()

        ## prepare an advertisement
        ## ..by exploration
        if round<exploration_round:
            offered_ad = random.choices(Ad.AllArms)[0]
            action_index = Ad.AllArms.index(offered_ad)
        ## ..by exploitation
        else:
            list_action_value = model.predict(np.array([context_array(user.group)]),verbose=0)[0]
            action_index = np.array(list_action_value).argmax()
            offered_ad = Ad.AllArms[action_index]

        ## will the user click?
        if user.will_click(offered_ad):
            num_clicks += 1
            reward = 1
        else:
            reward = 0

        ## keep the outcome in the memory
        memory.append((user.group,action_index,reward))

        ## do training when episode ends
        if len(memory)==episode_len:

            ## summarize the memory
            context_action_episode.clear()
            for (context,action,reward) in memory:
                context_action_life.update_reward(context,action,reward)
                context_action_episode.update_reward(context,action,reward)

            ## construct the data record for training
            x_train = []  # input contexts
            y_train = []  # output action values
            for context in range(num_contexts):
                x_train.append(context_array(context))
                y_value = model.predict(np.array([context_array(context)]),verbose=0)[0]
                for action in range(num_actions):
                    ## there are many ways we can incorporate the memory into FFNN
                    ## here we use exponential smoothing to incorporate the memory for
                    ## each episode into FFNN
                    y_value[action] = (1-alpha)*y_value[action] \
                                      + alpha*context_action_episode.get_mean_value(context,action,y_value[action])
                y_train.append(y_value)

            ## train the model
            model.fit(np.array([x_train]), np.array([y_train]), epochs=5, verbose=0)

            ## print the results
            print(" "*13,"Fact"," "*16,"This episode",
                  " "*12,"C-MAB table"," "*10,"FFNN prediction")
            for context in range(num_contexts):
                print(f"c={context}: ",end="")
                array_true    = [Ad.Type[arm][context] for arm in Ad.AllArms]
                array_episode = []
                array_life = []
                array_ffnn = []
                for action in range(num_actions):
                    array_life.append(context_action_life.get_mean_value(context,action,None))
                    array_episode.append(context_action_episode.get_mean_value(context,action,None))
                    array_ffnn = model.predict(np.array([context_array(context)]),verbose=0)[0]
                print(array_to_str(array_true),end="")
                print(array_to_str(array_episode),end="")
                print(array_to_str(array_life),end="")
                print(array_to_str(array_ffnn))
            memory.clear()
