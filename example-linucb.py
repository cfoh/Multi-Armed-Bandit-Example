'''
Linear UCB (LinUCB) is a contextual multi-armed bandit (CMAB) algorithm.
It assumes that the reward of each arm is a linear function of the context 
or feature vector. Its process is to learn the coefficients of the 
linear function for each arm, and then use the learned coefficients 
to select the best arm. The learned coefficients are estimated 
through Ridge regression.

In this example, we will use LinUCB to solve a digital advertising problem.
We want to maximize the click rate of advertisements by
selecting the best advertisement for each user.
'''

from math import ceil
import random
import matplotlib.pyplot as plt

from mab import LinUCB, EpsilonGreedy, ExplorationFirst

######################################################################
## User behaviour matrix for the environment (static class)
######################################################################

class Ad:

    ## ground truth click rate matrix
    Type = {       #  age group
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
        '''pick a random ad type.'''
        return random.choices(Ad.AllArms)[0]
    
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
    num_users = 10000    # number of users to visit the website

    ## run LinUCB
    num_features = len(Client().get_feature()) # 1 feature (age)
    linucb = LinUCB(num_features)

    ## setup exploration-exploitation strategy (pick one)
    strategy = EpsilonGreedy(0.15)
    #strategy = EpsilonGreedy(1.0) # set to 1.0 for 100% exploration
    #strategy = ExplorationFirst(0.2*num_users) # 20% exploration first

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    print(f"Testing {linucb.description()}")
    for round in range(num_users):

        ## a user has visited the website
        user = Client()
        this_feature = user.get_feature()
        this_reward = 0

        ## prepare an advertisement type
        ## ..by exploration
        if strategy.is_exploration(round):
            this_ad = Ad.pick_random_ad()
        ## ..by exploitation
        else:
            (this_ad,_) = linucb.get_best_arm(this_feature)
            if this_ad is None: # no info about this arm?
                this_ad = Ad.pick_random_ad()

        ## show the user 20 ads of the prepared type, 
        ## and check how many times will the user click
        for _ in range(20):
            if user.will_click(this_ad):
                this_reward += 1
        this_reward /= 20   # average click rate
        linucb.update_reward(arm=this_ad, reward=this_reward, context=this_feature)

        ## show progress bar
        bar_len = 50
        if round%(num_users/bar_len)==0:
            progress = ceil(round/num_users*bar_len)
            bar = "[" + "="*progress + ">" + " "*(bar_len-progress) + "]"
            print(f"Round {round}/{num_users}...{bar}", end="\r")

    print(f"{' '*bar_len} {' '*40}")

    ## prepare (x,y_true) and (x,y_pred) for plotting
    xi, yi, y_pred = {},{},{}
    for arm in Ad.AllArms:
        xi[arm] = []      # feature inputs
        yi[arm] = []      # true click rates
        y_pred[arm] = []  # predicted click rates
        start,stop = Ad.age_range()
        for age in range(start, stop+1):
            user_feature = Client(age).get_feature()
            xi[arm].append([age])
            yi[arm].append(Ad.expected_click_rate(arm, age))
            y_pred[arm].append(linucb.ridge_regression[arm].predict(user_feature))

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
    for age in range(start, stop):
        user_feature = Client(age).get_feature()
        best_arm = Ad.best_arm(age)
        expected_reward = Ad.expected_click_rate(best_arm, age)
        suggested_arm,_ = linucb.get_best_arm(user_feature)
        pred_reward = "[" + ", ".join(f"{linucb.get_reward(arm,user_feature):.2f}" for arm in Ad.AllArms) + "]"
        print(f"| {age:^3} | {best_arm:^7} | {expected_reward:^10.2f} | {suggested_arm:^9} | "
              f"{pred_reward:^{clen}} |" + ("  missed" if best_arm!=suggested_arm else ""))

    ## print bottom line
    print(f"+-----+---------+-------------------------{'-'*clen}--+")
