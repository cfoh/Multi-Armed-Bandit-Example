'''
In **Digital advertising**, we often see different users having
different preferences. It is thus inefficient to apply the 
same strategy to all users. Being able to differentiate users
into different `profiles` will enable the ML agent to treat 
users with different profiles differently. In this case, we
want to capture the `profiles` or `contexts`.

To illustrate the concept, we create an environment with 5 
user groups based on their age group. The following table
shows the likelihood of users in each age group clicking 
different types of advertisements.

```
  The Environment
+-------------------+--------------------------------------+
|                   |              Age group               |
| Ad Type           |  <25    26-35   36-45   46-55  >55   |
+-------------------+--------------------------------------+
| Toys & Games      |  80%     15%     10%     5%     5%   |
| Cars              |   5%     50%     30%    15%    10%   |
| Sports            |  15%     30%     40%    30%    30%   |
| Holiday Packages  |   5%     20%     35%    50%    50%   |
| Foods & Health    |   5%     25%     25%    40%    60%   |
+-------------------+--------------------------------------+
```

We shall use Contextual MAB to dea with this problem. 
For the sake of explanation, we introduce some simplified assumptions. 
- We use a small set of discrete contexts. 
- The contexts are independent of each other, meaning that 
  there is no relationship between the user behaviour any pair of 
  age groups. 
- When a user visits our website, we're able to somehow obtain the 
  user profile (or the context). 
  
With this very simple setup, we're able to design a simple discrete 
Contextual MAB learning algorithm. Our setup is:
- the `context` is the age group
- the `arms` are advertisement type to offer
- the `reward` is 1 if a click is registered, 0 otherwise
'''

from math import ceil
import random
import time
import matplotlib.pyplot as plt

from mab import MAB, CMAB

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
    AgeGroupSize = len(AllArms)
    AllAgeGroups = range(AgeGroupSize)

######################################################################
## Theoretical result calculator (static class)
######################################################################

class Theoretical:

    best_arm = {}

    def expected_click_rate(arm,context) -> float:
        '''This is commonly notated as $\mu(a)$.'''
        return Ad.Type[arm][context]

    def optimal_click_rate(context) -> float: 
        '''This is commonly notated as $\mu^*$, which is
        $\max_{a\in A} \mu(a)$ for a specific context.
        '''
        return max([mu_a[context] for mu_a in list(Ad.Type.values())])

    def optimal_arm(context):
        '''It returns the optimal arm based on the given `context`.'''
        if context in Theoretical.best_arm:
            return Theoretical.best_arm[context]
        the_best_arm = { "arm":None, "value":0 }
        for arm in Ad.AllArms:
            if Ad.Type[arm][context]>=the_best_arm["value"]:
                the_best_arm["arm"] = arm
                the_best_arm["value"] = Ad.Type[arm][context]
        Theoretical.best_arm[context] = the_best_arm["arm"]
        return Theoretical.best_arm[context]

    def overall_optimal_click_rate() -> float: 
        '''This is the overall optimal click rate across all user groups.'''
        click_rate = []
        for context in Ad.AllAgeGroups:
            click_rate.append(Theoretical.optimal_click_rate(context))
        return sum(click_rate)/len(click_rate) # assume user groups appear equally


######################################################################
## MAB empirical result keeper 
######################################################################

class Empirical:

    def __init__(self):
        ## data series
        self.no_regrets = []       # store the history of no regret count
        self.click_selections = [] # store the history of click selections
        self.click_outcomes = []   # store the history of click outcomes
        self.click_context = []    # store the history of contexts
        self.count_selection = {}  # store the total count of each arm selection

    def report(self, arm, outcome, context):
        self.click_outcomes.append(outcome)
        self.click_selections.append(arm)
        self.click_context.append(context)
        if arm not in self.count_selection:
            self.count_selection[arm] = 0
        else:
            self.count_selection[arm] += 1
        no_regret = 1 if arm==Theoretical.optimal_arm(context) else 0
        if len(self.no_regrets)==0:
            self.no_regrets.append(no_regret)
        else:
            self.no_regrets.append(self.no_regrets[-1]+no_regret)

    def get_arm_count(self, arm):
        if arm not in self.count_selection:
            return 0
        return self.count_selection[arm]

    def get_click_rate(self):
        return sum(self.click_outcomes)/len(self.click_outcomes)

    def get_hit_count(self):
        return self.no_regrets[-1]

    def get_click_rate_series(self):
        click_rate_series = []
        click_rate_total = 0
        click_rate_size = 0
        for click in self.click_outcomes:
            click_rate_total += 1 if click else 0
            click_rate_size += 1
            click_rate_series.append(click_rate_total/click_rate_size)
        return click_rate_series

    def get_arm_selection_series(self):
        arm_selection_series = {}
        for arm in Ad.AllArms:
            arm_selection_series[arm] = [0]
        for selected_arm in self.click_selections:
            for arm in Ad.AllArms:
                if arm==selected_arm:
                    arm_selection_series[arm].append(arm_selection_series[arm][-1]+1)
                else:
                    arm_selection_series[arm].append(arm_selection_series[arm][-1])
        for arm in Ad.AllArms:
            arm_selection_series[arm] = arm_selection_series[arm][1:]
        return arm_selection_series


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
        click_prob = random.randint(0,99)
        if click_prob<100*Ad.Type[ad][self.group]:
            return True
        return False


####################################################################
## MAB Strategy
####################################################################

class BaseStrategy:
    def description(self):
        return f"100% exploration"
    def is_exploration(self,round):
        return True # default is 100% exploration

class EpsilonGreedy(BaseStrategy):
    def __init__(self,epsilon):
        self.epsilon = epsilon
    def description(self):
        return f"Epsilon Greedy, epsilon = {self.epsilon}"
    def is_exploration(self,round):
        return random.random()<self.epsilon

class ExplorationFirst(BaseStrategy):
    def __init__(self,switch_round):
        self.switch_round = int(switch_round)
    def description(self):
        return f"Exploration first for {self.switch_round} rounds"
    def is_exploration(self,round):
        return round<self.switch_round

####################################################################
## main loop
####################################################################

if __name__ == "__main__":

    ## setup environment parameters
    num_users = 10000    # number of users to visit the website
    mab_num_clicks = 0   # number of clicks collected for mab
    cmab_num_clicks = 0  # number of clicks collected for cmab

    ## we run both agents together
    mab  = MAB()      # simple MAB agent
    cmab = CMAB()     # CMAB agent
    mab_out = Empirical()
    cmab_out = Empirical()

    ## setup exploration-exploitation strategy (pick one)
    strategy = EpsilonGreedy(0.15)
    #strategy = EpsilonGreedy(1.0) # set to 1.0 for 100% exploration
    #strategy = ExplorationFirst(0.2*num_users) # 20% exploration first

    ## ready-set-go
    print("\n")
    spiner = ["\u2212","\\","|","/","\u2212","\\","|","/"]
    for i in range(40,0,-1):
        print(f"\033[KRunning in ...{ceil(i/10)} {spiner[i%len(spiner)]}")
        print("\033[2A")
        time.sleep(0.1)
    print(f"\033[K")

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    print(f"Testing {cmab.description()}\n")
    for round in range(num_users):

        ## a user has visited the website
        user = Client()

        ## prepare an advertisement
        ## ..by exploration
        if strategy.is_exploration(round):
            mab_ad = cmab_ad = random.choices(Ad.AllArms)[0]
        ## ..by exploitation
        else:
            ## for mab
            (mab_ad,_) = mab.get_best_arm()
            if mab_ad is None: # no info about this arm?
                mab_ad = random.choices(Ad.AllArms)[0]
            ## for cmab
            (cmab_ad,_) = cmab.get_best_arm(user.group)
            if cmab_ad is None: # no info about this user group?
                cmab_ad = random.choices(Ad.AllArms)[0]

        ## for mab, will the user click?
        if user.will_click(mab_ad):
            mab_num_clicks += 1
            reward = 1
        else:
            reward = 0
        mab.update_reward(arm=mab_ad, reward=reward)
        mab_out.report(mab_ad, reward, user.group)

        ## for cmab, will the user click?
        if user.will_click(cmab_ad):
            cmab_num_clicks += 1
            reward = 1
        else:
            reward = 0
        cmab.update_reward(arm=cmab_ad, reward=reward, context=user.group)
        cmab_out.report(cmab_ad, reward, user.group)

        ## show animation 
        mab_no_regret = int(60*mab_out.get_hit_count()/(round+1))
        cmab_no_regret = int(60*cmab_out.get_hit_count()/(round+1))
        print(f"\033[KNumber of visitors = {round+1}")
        print(f"\033[KNumber of optimal arms played:")
        print(f"\033[K> MAB  [%s] %d"%("="*mab_no_regret,mab_out.get_hit_count()))
        print(f"\033[K> CMAB [%s] %d"%("="*cmab_no_regret,cmab_out.get_hit_count()))
        print("\033[5A")
        time.sleep(0.05 if round<150 else 0.01 if round<2000 else 0.001)

    ## show outcome
    mab_average_click_rate  = mab_num_clicks/num_users
    cmab_average_click_rate = cmab_num_clicks/num_users
    best_click_rate = Theoretical.overall_optimal_click_rate()
    print("%s"%"\n"*4)
    print(f"Strategy: {strategy.description()}")
    print(f"Number of users = {num_users}")
    print(f"Theoretical best click rate = {100*best_click_rate:4.1f}%\n")
    print(f"                     MAB   CMAB ")
    print(f"                   -------------")
    print(f"Number of clicks = {mab_num_clicks:>5d}  {cmab_num_clicks:>5d}")
    print(f"Click rate       = {100*mab_average_click_rate:3.1f}%  "
                           + f"{100*cmab_average_click_rate:3.1f}%")
    print()

    ## plot no regret evolution
    plt.figure(1)
    plt.plot(range(len(mab_out.no_regrets)), mab_out.no_regrets, 'r-', label="MAB")
    plt.plot(range(len(cmab_out.no_regrets)), cmab_out.no_regrets, 'b-', label="CMAB")
    plt.xlabel("Number of ads offered")
    plt.ylabel("No Regret Count")
    plt.legend(loc="upper left") 
    plt.show()