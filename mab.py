'''
**Digital advertising** is a form of marketing that targets online
users. A simple example of online marketing is where a website
embeds a small advertisement banner with the objective that 
users visiting the website will click the advertisement banner 
to explore the advertised products or services.

However, different users have different interests, and thus not
all types of advertisements will attract all users. To achieve
an effective online advertisement, it is necessary to know
the user group and how they respond to each type of advertisements.
In other words, we need to find the relationship between user 
group and the type of advertisements that interests them most.

In this tutorial, we shall use **Multi-Armed Bandit** (MAB) 
reinforcement learning (RL) to establish the relationship. MAB
is an online learning meaning that the machine learning (ML) agent
learns during the operation. 

In reinforcement learning, the ML agent interacts with the environment 
to learn the behaviour of the environment and try to find appropriate
actions that can maximize the rewards. The following diagram illustrates
the interaction between the ML agent and the environment:
```
        +-----------------------+
        |        Action         |
        |                       V
   +---------+           +--------------+
   |   ML    |           |              |
   |  Agent  |           | Environment  |
   +---------+           |              |
        ^                +--------------+
        |    State, Reward      |
        +-----------------------+
```

We first need to define the behaviour of users. This is the 
`environment` defining how a user responds to an offered 
advertisement. The user behaviour is not known to the ML agent,
and it is the task of the ML agent to learn and establish the
relationship. The ML agent knows the user profile, and its job
is to discover which advertisement is most attactive to the
user.

The behaviour of users are described in the following table. 
It shows the likelihood of each age clicking different types
of advertisements.
```
  The Environment
+-------------------+--------------------------------------+
|                   |              Age group               |
| Ad Type           |  <25    26-35   36-45   46-55  >55   |
+-------------------+--------------------------------------+
| Toys & Games      |  80%     15%      5%     5%     5%   |
| Cars              |   5%     50%     15%    10%     5%   |
| Sports            |   5%     10%     40%    25%    10%   |
| Holiday Packages  |   5%     20%     20%    50%    20%   |
| Foods & Health    |   5%      5%     20%    10%    60%   |
+-------------------+--------------------------------------+
```

With Contextual MAB, our setup is:
- the `context` is the age group
- the `arms` are advertisement type to offer
- the `reward` is 100 if click is registered, otherwise 0
'''

import operator
import random
import matplotlib.pyplot as plt

######################################################################
## Multi-Armed Bandit & Contextual MAB
######################################################################
class MAB:
    '''Simple Multi-armed Bandit implementation.'''
    def __init__(self):
        self.total_rewards = {}
        self.total_count = {}
        self.average_reward = {}

    def update_reward(self, arm, reward):
        if arm not in self.total_rewards: 
            self.total_rewards[arm] = 0
            self.total_count[arm] = 0
        self.total_rewards[arm] += reward
        self.total_count[arm] += 1
        self.average_reward[arm] = self.total_rewards[arm]/self.total_count[arm]

    def get_reward(self, arm):
        if arm not in self.average_reward: return 0
        return self.average_reward[arm]

    def get_best_arm(self): # return a tuple (arm,reward)
        return max(self.average_reward.items(), key=operator.itemgetter(1))
 

class CMAB:
    '''Simple Contextual Multi-armed Bandit implementation.'''
    def __init__(self):
        self.mab = {}

    def update_reward(self, arm, reward, context=None):
        if context not in self.mab: self.mab[context] = MAB()
        self.mab[context].update_reward(arm, reward)

    def get_reward(self, arm, context=None):
        if context not in self.mab: return 0
        return self.mab[context].get_reward(arm)

    def get_best_arm(self, context=None): # return a tuple (arm,reward)
        if context not in self.mab: return (None,None)
        return self.mab[context].get_best_arm()

######################################################################
## User behaviour matrix (static class)
######################################################################

class Ad:

    AgeGroupSize = 5
    Type = {    ##   <25 <35 <45 <55 >55  (age group)
        "toys"     : [80, 15,  5,  5,  5],
        "cars"     : [ 5, 50, 15, 10,  5],
        "sports"   : [ 5, 10, 40, 25, 10],
        "holidays" : [ 5, 20, 20, 50, 20],
        "foods"    : [ 5,  5, 20, 10, 60]
    }
    ListAdType = list(Type.keys()) # list of all ad types
    ListAgeGroup = range(AgeGroupSize)

    @staticmethod
    def check():
        ## perform checking
        for age_group in Ad.ListAgeGroup:
            percentage = 0
            for ad_type in Ad.Type:
                percentage += Ad.Type[ad_type][age_group]
            if percentage!=100:
                print(f"Warning: Age group {age_group} doesn't add up to 100%")
        for ad_type in Ad.Type:
            if len(Ad.Type[ad_type])!=Ad.AgeGroupSize:
                print(f"Warning: Ad type {ad_type} doesn't contain all age groups")


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
        '''Will this client clicks this advert?'''
        ad_types = Ad.ListAdType
        ad_weights = [w[self.group] for w in list(Ad.Type.values())]
        if ad in random.choices(ad_types,ad_weights):
            return True
        return False

######################################################################
## Click results (static class) for plotting
######################################################################

class ClickResult:

    Group = [[] for i in range(Ad.AgeGroupSize)]

    @staticmethod
    def report(group, click):
        ClickResult.Group[group].append(click)

    @staticmethod
    def get_click_rate(group):
        click_rate_list = []
        click_rate_total = 0
        click_rate_size = 0
        for click in ClickResult.Group[group]:
            click_rate_total += 1 if click else 0
            click_rate_size += 1
            click_rate_list.append(click_rate_total/click_rate_size)
        return click_rate_list


####################################################################
## main launcher
####################################################################

if __name__ == "__main__":

    ## check the user behaviour matrix
    Ad.check()

    ## setup test parameters
    cmab = CMAB()  # Contextual MAB agent
    num_users = 10000 # number of users to visit the website
    num_clicks = 0    # number of clicks collected

    ## set epsilon, i.e. the percentage of exploration
    epsilon = 0.15 
    #epsilon = 1.0  # set to 1.0 for 100% exploration

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    for round in range(num_users):

        ## a user has visited the website
        user = Client()

        ## prepare an advertisement
        ## ..by exploration
        if random.random()<epsilon:
            offered_ad = random.choices(Ad.ListAdType)[0]
        ## ..by exploitation
        else:
            (offered_ad,reward) = cmab.get_best_arm(user.group)
            if offered_ad is None: # no info about this user group?
                offered_ad = random.choices(Ad.ListAdType)[0]

        ## will the user click?
        if user.will_click(offered_ad):
            click_reward = 100
            num_clicks += 1
            ClickResult.report(user.group,True)
        else:
            click_reward = 0
            ClickResult.report(user.group,False)
        cmab.update_reward(arm=offered_ad, reward=click_reward, context=user.group)

    ## show outcome
    print(f"Epsilon = {epsilon}")
    print(f"Number of users = {num_users}")
    print(f"Number of clicks = {num_clicks}")
    print(f"Click rate = {100*num_clicks/num_users:1.2f}%")

    ## analyze theoretical best outcome
    ## this is derived as if we know the most attractive ad
    ## to the user group and always offer those ad to the users
    best_click_rate = 0
    for group in Ad.ListAgeGroup:
        best_click_rate += max([w[group] for w in list(Ad.Type.values())])
    best_click_rate /= Ad.AgeGroupSize
    print(f"Theoretical best click rate = {best_click_rate:1.2f}%\n")

    ## analyze CMAB
    print(f"Age group   Ad to offer    Expected CMAB reward")
    print(f"===============================================")
    for group in Ad.ListAgeGroup:
        (offered_ad,reward) = cmab.get_best_arm(group)
        print(f"    {group}"," "*7,f"{offered_ad}"," "*(16-len(offered_ad)),f"{reward:1.2f}")

    ## plot the click rate for each age group
    line_pattern = { 0: "r-", 1: "g-", 2: "b-", 3: "c-", 4: "k-", }
    for group in Ad.ListAgeGroup:
        this_series = ClickResult.get_click_rate(group)
        plt.plot(range(len(this_series)), this_series, 
                    line_pattern[group], label=f"Age group {group}")
    plt.legend()
    plt.xlabel("Number of users visited the website")
    plt.ylabel("Click rate")
    plt.show()
