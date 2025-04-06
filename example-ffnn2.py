'''
This is an example of using a Feed Forward Neural Network (FFNN) to solve
a Contextual Multi-Armed Bandit (C-MAB) problem. The FFNN is used to predict 
the click through rate of different ads based on the user's age. The MAB 
problem is solved using an epsilon-greedy strategy, where the agent explores 
different ads with a certain probability (epsilon) and exploits the best ad 
with the highest predicted click through rate.

The main task of C-MAB is to predict the click through rate of different ads
for different users based on their features (in this case, age). In other 
words, the agent needs to learn the mapping between user features and the
click through rate of different ads. FFNN is used to learn this mapping.
'''


import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mab import EpsilonGreedy, ExplorationFirst

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

######################################################################
## User behaviour matrix for the environment (static class)
######################################################################

class Ad:

    ## ground truth click rate matrix
    Type = {
      #  arm       #  <25   <35   <45    <55  >=55
      #  --------------------------------------------
        "toys"     : [0.76, 0.64, 0.52, 0.40, 0.28],
        "cars"     : [0.10, 0.25, 0.40, 0.55, 0.70],
        "sports"   : [0.15, 0.30, 0.70, 0.30, 0.15],
        "foods"    : [0.05, 0.25, 0.25, 0.40, 0.80]
    }
    ## in our example, the best ad is:
    ## - Toys for age < 34
    ## - Sports for age between 35 and 44
    ## - Cars for age between 45 and 54
    ## - Food for age >= 55

    AllArms = list(Type.keys()) # list of all ad types
    AgeGroupRange = [(5,24), (25,34), (35,44), (45,54), (55,70)]
    AgeGroupSize = len(AgeGroupRange)
    AllAgeGroups = range(AgeGroupSize)

    @staticmethod
    def pick_random_ad():
        return random.choice(list(Ad.Type.keys()))

    @staticmethod
    def age_to_group(age):
        '''return the age group of a specific age.'''
        return next((i for i,r in enumerate(Ad.AgeGroupRange) if r[0]<=age<=r[1]), None)

    @staticmethod
    def expected_click_rate(arm, age):
        '''return the click rate of a specific arm for a specific age group.'''
        return Ad.Type[arm][Ad.age_to_group(age)]

    @staticmethod
    def best_arm(age):
        '''return the best arm for a specific age group.'''
        return max(Ad.AllArms, key=lambda x: Ad.expected_click_rate(x, age))

    @staticmethod
    def age_range():
        '''return the age range (min_age,max_age).'''
        all_age_values = [value for t in Ad.AgeGroupRange for value in t] # flatten the list of age range
        return (min(all_age_values),max(all_age_values))

######################################################################
## Client profile
######################################################################

class Client:
    def __init__(self, age=None):
        '''create a user with a random age group or 
        assign a specific age group to this new user.'''
        if age is None:
            ## pick a random age group
            self.group = random.choice(Ad.AllAgeGroups)
            ## pick a random age within the group
            low_age, high_age = Ad.AgeGroupRange[self.group]
            self.age = random.randint(low_age, high_age) # then pick a random age within
        else:
            self.age = age
            self.group = Ad.age_to_group(age)

    def get_feature(self):
        '''return the feature vector (as a tuple) of this user.'''
        return (self.age,)

    def will_click(self, ad) -> bool:
        '''Will this client click this advert?'''
        click_prob = random.randint(0,99)
        if click_prob<100*Ad.Type[ad][self.group]:
            return True
        return False
    
####################################################################
## main loop
####################################################################

if __name__ == "__main__":

    ## setup environment parameters
    random.seed(45)      # set random seed for reproducibility
    num_users = 50000    # number of users to visit the website

    ## setup training arrangement
    episode_len = 1000        # the length of an episode for each training
    exploration_round = 8000  # explore so many rounds first
    exploration_count = 0     # the number of explorations
    memory_data = []          # the memory to keep feature-reward pairs
    memory_len = 0            # the length of total data in memory_data

    ## setup exploration-exploitation strategy (pick one)
    strategy = EpsilonGreedy(0.15)
    #strategy = EpsilonGreedy(1.0) # set to 1.0 for 100% exploration
    #strategy = ExplorationFirst(0.2*num_users) # 20% exploration first

    ## create FFNN model, input_dim=1 (i.e. age), output_dim=4 (i.e. 4 ads)
    num_features = len(Client().get_feature()) # 1 feature (age)
    num_rewards  = len(Ad.AllArms)             # 4 ads
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=num_features))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_rewards, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')

    ## model prediction cache to avoid repeated prediction within the same episode
    prediction_cache = {}

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    print(f"Testing Deep Learning MAB")
    for round in range(1,num_users+1):

        ## a user has visited the website
        user = Client()
        this_feature = user.get_feature()
        this_reward = 0

        ## prepare an advertisement
        ## ..by exploration
        if round<exploration_round or strategy.is_exploration(round):
            this_ad = Ad.pick_random_ad()
            action_index = Ad.AllArms.index(this_ad)
            exploration_count += 1
        ## ..by exploitation
        else:
            if this_feature in prediction_cache: # cache hit?
                reward_list = prediction_cache[this_feature]
            else:
                reward_list = model.predict(np.array([this_feature]), verbose=0)
                prediction_cache[this_feature] = reward_list
            max_arm_index = np.argmax(reward_list, axis=1)
            this_ad = Ad.AllArms[max_arm_index[0]]

        ## show the user 20 ads of the prepared type, 
        ## and check how many times will the user click
        for _ in range(20):
            if user.will_click(this_ad):
                this_reward += 1
        this_reward /= 20   # average click rate

        ## store (feature,ad,reward) tuple in memory_data
        memory_data.append((this_feature, this_ad, this_reward))

        ## train the model at the end of an episode
        if len(memory_data)==episode_len:

            ## the first step is to reformat memory_data making it
            ## suitable for training:
            ##     memory_data                     train_data
            ## (feature,arm,reward)       (feature,list_of_all_rewards)
            ## ------------              -------------
            ## [((age1,),'toys',toy1),    [((age1,),[toy1,car_pred,sport_pred,food_pred]),
            ##  ((age2,),'foods',food2),   ((age2,),[toy_pred,car_pred,sport_pred,food2]),
            ##     ...                      ...
            ##  ((age_k,),'cars',car_k)]   ((age_k,),[toy_pred,car_k,sport_pred,food_pred])]
            ##
            ## since train_data needs the full list of rewards for each
            ## data point, but memory_data only provides one specific 
            ## reward, we fill the missing rewards by using existing partially
            ## trained model to do prediction.

            ## perform prediction for the missing rewards
            X = np.array([list(item[0]) for item in memory_data], dtype=np.float32)
            y = model.predict(X, verbose=0)  # shape: (N,4)

            ## create a dictionary called `prediction` mapping each 
            ## feature (xi) to each prediction (yi_pred) for easy access
            ## to predicted rewards
            prediction = {
                tuple(xi): yi_pred.tolist() for xi, yi_pred in zip(X, y)
            }

            ## create train_data by combining memory_data and 
            ## predicted missing rewards
            train_data = []
            for this_feature, this_ad, this_reward in memory_data:
                this_reward_list = prediction[this_feature]
                this_reward_list[Ad.AllArms.index(this_ad)] = this_reward
                train_data.append((this_feature, this_reward_list))

            ## prepare data (X,y) and perform the training
            X = np.array([list(d[0]) for d in train_data], dtype=np.float32)
            y = np.array([d[1] for d in train_data], dtype=np.float32)
            model.fit(X, y, batch_size=30, epochs=5, verbose=0)

            ## clear memory_data and prediction_cache for the next episode
            memory_data.clear()
            prediction_cache = {}

            ## show progress
            print(f"Round {round}/{num_users}, exploration rate = {exploration_count/episode_len:.2f}")
            exploration_count = 0

    ## prepare (x,y_true) and (x,y_pred) for plotting
    xi, yi, y_pred = {},{},{}
    for arm in Ad.AllArms:

        ## prepare storages
        xi[arm] = []      # feature inputs
        yi[arm] = []      # true click rates
        y_pred[arm] = []  # predicted click rates
        start,stop = Ad.age_range()

        ## iterate all features and predict the rewards
        feature_list = [Client(age).get_feature() for age in range(start, stop+1)]
        predictions = model.predict(np.array(feature_list), verbose=0)

        ## create a dictionary mapping each feature to its predicted rewards
        ## for a specific arm (or ad)
        ad_predictions = predictions[:, Ad.AllArms.index(arm)]
        age_score_dict = {
            age: float(score) for age, score in zip(range(start,stop+1), ad_predictions)
        }

        ## fill in the xi,yi,y_pred
        for age in range(start, stop+1):
            xi[arm].append([age])
            yi[arm].append(Ad.expected_click_rate(arm, age))
            y_pred[arm].append(age_score_dict[age])

    ## plot xi,yi,y_pred
    plt.figure()
    for arm in Ad.AllArms:
        plt.plot(xi[arm], yi[arm], label=arm)
        plt.plot(xi[arm], y_pred[arm], linestyle='--', label=f"{arm} prediction")
    plt.xlabel('Age')
    plt.ylabel('Click Rate')
    plt.title('Click Rate vs Age')
    plt.legend()
    plt.show()

    ## write all ads in a string
    ad_str = "[" + ", ".join(str(arm) for arm in Ad.AllArms) + "]"
    clen = len(ad_str)
        
    ## print table header
    print(f"+-----------------------------------------{'-'*clen}--+")
    print(f"| age |   best  |  expected  | suggested | {'predicted click rate':^{clen}} |")
    print(f"+     |    ad   | click rate |     ad    | {ad_str} |")
    print(f"+-----+---------+-------------------------{'-'*clen}--+")

    ## print table content
    for age in range(start, stop+1):
        ## best arm based on ground truth
        best_arm = Ad.best_arm(age)
        expected_reward = Ad.expected_click_rate(best_arm, age)

        ## best arm based on model prediction
        reward_list = model.predict(np.array([Client(age).get_feature()]), verbose=0)
        max_arm_index = np.argmax(reward_list, axis=1)
        suggested_arm = Ad.AllArms[max_arm_index[0]]
        pred_reward = "[" + ", ".join(f"{reward:.2f}" for reward in reward_list[0]) + "]"

        ## print the table content
        print(f"| {age:^3} | {best_arm:^7} | {expected_reward:^10.2f} | {suggested_arm:^9} | "
              f"{pred_reward:^{clen}} |" + ("  missed" if best_arm!=suggested_arm else ""))

    ## print bottom line
    print(f"+-----+---------+-------------------------{'-'*clen}--+")
