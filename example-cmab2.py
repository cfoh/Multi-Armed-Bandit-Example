'''
In this example, we summarize both user profiles and actions to 
form contexts. We can then use MAB to perform learning. Some
changes are needed. Briefly, the algorithm can be described by:
- Observe the user profile and produce context for each action
- Rank the produced contexts based on their average reward
- Pick the best context. The action associated with the context
  will be executed
- Observe the reward after executing the action, update the 
  context based on the observed reward
'''

from math import ceil
import random
import time

from mab import CMAB2
from mab import ExplorationFirst, EpsilonGreedy, EpsilonDecreasing

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
    AgeGroupSize = len(list(Type.values())[0])
    AllAgeGroups = range(AgeGroupSize)

######################################################################
## Theoretical result calculator (static class)
######################################################################

class Theoretical:

    best_arm = {}

    @staticmethod
    def expected_click_rate(arm,context) -> float:
        '''This is commonly notated as $\mu(a)$.'''
        return Ad.Type[arm][context]

    @staticmethod
    def optimal_click_rate(context) -> float: 
        '''This is commonly notated as $\mu^*$, which is
        $\max_{a\in A} \mu(a)$ for a specific context.
        '''
        return max([mu_a[context] for mu_a in list(Ad.Type.values())])

    @staticmethod
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

    @staticmethod
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
    num_users = 10000    # number of users to visit the website
    num_clicks = 0       # number of clicks collected for cmab
    animation  = True   # True/False

    ## we the agent
    cmab = CMAB2()    # MAB agent with summarized contexts
    cmab_out = Empirical()

    ## setup exploration-exploitation strategy (pick one)
    strategy = EpsilonGreedy(0.15)
    #strategy = EpsilonDecreasing(-0.5)
    #strategy = EpsilonGreedy(1.0) # set to 1.0 for 100% exploration
    #strategy = ExplorationFirst(0.2*num_users) # 20% exploration first
    #strategy = ExplorationFirst(0.02*num_users) # 2% exploration first

    ## ready-set-go
    print("\n")
    spinner = ["\u2212","\\","|","/","\u2212","\\","|","/"]
    for i in range(40,0,-1):
        print(f"\033[KRunning in ...{ceil(i/10)} {spinner[i%len(spinner)]}")
        print("\033[2A")
        time.sleep(0.1*animation)
    print(f"\033[K")

    ## animation content
    print(f"Testing {cmab.description()}\n")
    print(f"Number of ads presented:")
    print(f"Age Group:  {1:5d} {2:5d} {3:5d} {4:5d} {5:5d}")
    print(f"            {'-'*30}")
    count = {}
    for ad in Ad.AllArms:
        count[ad] = [0]*Ad.AgeGroupSize

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    for round in range(num_users):

        ## a user has visited the website
        user = Client()

        ## prepare an advertisement
        ## ..by exploration
        if strategy.is_exploration(round):
            offered_ad = random.choices(Ad.AllArms)[0]
        ## ..by exploitation
        else:
            (offered_ad,_) = cmab.get_best_arm(cmab.context(user.group))
            if offered_ad is None: # no info about this user group?
                offered_ad = random.choices(Ad.AllArms)[0]

        ## will the user click?
        if user.will_click(offered_ad):
            num_clicks += 1
            reward = 1
        else:
            reward = 0
        context = cmab.context(user.group,offered_ad)
        cmab.update_reward(context, reward)
        cmab_out.report(offered_ad, reward, user.group)

        ## show animation 
        count[offered_ad][user.group] += 1
        for ad in Ad.AllArms:
            print(f"\033[K  {ad:11s}",end="")
            for grp in Ad.AllAgeGroups:
                print(f"{count[ad][grp]:4d}  ",end="")
            print()
        print(f"\nNumber of visitors = {round}")
        if animation:
            time.sleep(0.05 if round<150 else 0.01 if round<2000 else 0.001)
        print("\033[8A")

    ## show outcome
    cmab_average_click_rate = num_clicks/num_users
    best_click_rate = Theoretical.overall_optimal_click_rate()
    print("%s"%"\n"*5)
    print(f"Strategy: {strategy.description()}")
    print(f"Number of users = {num_users}")
    print(f"Theoretical best click rate = {100*best_click_rate:4.1f}%\n")
    print(f"Number of clicks = {num_clicks:>5d}")
    print(f"Click rate       = {100*cmab_average_click_rate:3.1f}%")
    print()

