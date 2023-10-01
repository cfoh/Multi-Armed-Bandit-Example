'''
**Digital advertising** is a form of marketing that targets online
users. A simple example of online marketing is where a website
embeds a small advertisement banner with the objective that 
users visiting the website will click the advertisement banner 
to explore the advertised products or services.

However, not all types of advertisements will attract all users.
In our example, we have 5 types of ads we can put in the banner.
For each user visiting our webpage, we need to decide which 
type of ads we should show to the user such that we can achieve
the highest click through rate, i.e. the highest chance of users
clicking the presented ad. 

In this tutorial, we shall use **Multi-Armed Bandit** (MAB) 
reinforcement learning (RL) to perform decision making. MAB
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
relationship. Its job is to discover which advertisement is 
the most attractive to the user.

The behaviour of users are described in the following table. 
It shows the likelihood of users clicking different types of 
advertisements.

```
  The Environment
+-------------------+---------------+
|                   |  Probability  |
| Ad Type           |  of clicking  |
+-------------------+---------------+
| Toys & Games      |      10%      |
| Cars              |      30%      |
| Sports            |      40%      |
| Holiday Packages  |      35%      |
| Foods & Health    |      25%      |
+-------------------+---------------+
```

Our setup is:
- the `arms` are advertisement type to offer
- the `reward` is 1 if a click is registered, 0 otherwise
'''

from math import ceil
import random
import time
import matplotlib.pyplot as plt

from mab import MAB
from mab import ExplorationFirst, EpsilonGreedy, EpsilonDecreasing

######################################################################
## User behaviour matrix for the environment (static class)
######################################################################

class Ad:

    Type = {
      #  arm      expected reward 
      #  ------------------------
        "toys"     : 0.10,
        "cars"     : 0.30,
        "sports"   : 0.40,
        "holidays" : 0.35,
        "foods"    : 0.25
    }
    AllArms = list(Type.keys()) # list of all ad types


######################################################################
## Theoretical result calculator (static class)
######################################################################

class Theoretical:

    regret_series = [] # store the regret series

    @staticmethod
    def expected_click_rate(arm) -> float:
        '''This is commonly notated as $\mu(a)$.'''
        return Ad.Type[arm]

    @staticmethod
    def optimal_click_rate() -> float: 
        '''This is commonly notated as $\mu^*$, which is
        $\max_{a\in A} \mu(a)$
        '''
        return max([mu_a for mu_a in list(Ad.Type.values())])

    @staticmethod
    def regret(t) -> float:
        '''This is commonly notated as $R(T)$, which is the regret
        at round $T$. It is calculated by 
        $R(T) = T \mu^* - \sum_{t=1}^T \mu(a_t)$
        where $a_t$ is the arm selection history.
        '''
        optimal = Theoretical.optimal_click_rate() * t  # optimal click rate
        experienced = 0                             # experienced click rate
        for arm in Ad.AllArms:
            experienced += Theoretical.expected_click_rate(arm) * Empirical.get_arm_count(arm)
        regret_at_t = optimal - experienced
        Theoretical.regret_series.append(regret_at_t)
        return regret_at_t

    @staticmethod
    def get_regret_series():
        return Theoretical.regret_series


######################################################################
## Historical result keeper (static class)
######################################################################

class Empirical:

    click_selections = [] # store the history of click selections
    click_outcomes = []   # store the history of click outcomes
    count_selection = {}  # store the total count of each arm selection

    @staticmethod
    def report(arm,outcome):
        Empirical.click_outcomes.append(outcome)
        Empirical.click_selections.append(arm)
        if arm not in Empirical.count_selection:
            Empirical.count_selection[arm] = 0
        else:
            Empirical.count_selection[arm] += 1

    @staticmethod
    def get_arm_count(arm):
        if arm not in Empirical.count_selection:
            return 0
        return Empirical.count_selection[arm]

    @staticmethod
    def get_click_rate():
        return sum(Empirical.click_outcomes)/len(Empirical.click_outcomes)

    @staticmethod
    def get_click_rate_series():
        click_rate_series = []
        click_rate_total = 0
        click_rate_size = 0
        for click in Empirical.click_outcomes:
            click_rate_total += 1 if click else 0
            click_rate_size += 1
            click_rate_series.append(click_rate_total/click_rate_size)
        return click_rate_series

    @staticmethod
    def get_arm_selection_series():
        arm_selection_series = {}
        for arm in Ad.AllArms:
            arm_selection_series[arm] = [0]
        for selected_arm in Empirical.click_selections:
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

    def will_click(self, ad) -> bool:
        '''Will this client click this advert?'''
        click_prob = random.randint(0,99)
        if click_prob<100*Ad.Type[ad]:
            return True
        return False


####################################################################
## main loop
####################################################################

if __name__ == "__main__":

    ## setup environment parameters
    num_users = 2000 # number of users to visit the website
    num_clicks = 0   # number of clicks collected
    animation  = True # True/False

    ## setup MAB
    mab = MAB()       # simple MAB agent

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

    ## print heading for the animation
    print(f"Testing {mab.description()}\n")
    print("Ad_type   Reward  Ad_shown_to_users")
    print("-----------------------------------")

    ## this is the main loop
    ## the objective of ML agent is to achieve 
    ## as many clicks as possible through learning
    for round in range(1,num_users+1):

        ## a user has visited the website
        user = Client()

        ## prepare an advertisement
        ## ..either by exploration
        if strategy.is_exploration(round):
            offered_ad = random.choices(Ad.AllArms)[0]
        ## ..or by exploitation
        else:
            (offered_ad,reward) = mab.get_best_arm()
            if offered_ad is None: # no info about this arm yet?
                offered_ad = random.choices(Ad.AllArms)[0]

        ## will the user click?
        if user.will_click(offered_ad):
            click_reward = 1
            num_clicks += 1
        else:
            click_reward = 0
        Empirical.report(offered_ad, click_reward)
        mab.update_reward(arm=offered_ad, reward=click_reward)

        ## show animation
        for arm in Ad.AllArms:
            r = mab.get_reward(arm)
            len_count_bar = int(50*Empirical.get_arm_count(arm)/round)
            print(f"\033[K> {arm:8s} {r:5.2f} ",end="")
            print("*" if arm==offered_ad else " ",end="")
            print("[%s] %d"%("="*len_count_bar,Empirical.get_arm_count(arm)))
        current_click_rate = Empirical.get_click_rate()
        current_regret = Theoretical.regret(round)
        print(f"\nClick rate = {current_click_rate:5.2f}")
        print(f"Regret = {current_regret:5.2f}")
        print("\033[9A")
        time.sleep(0.05*animation if round<1000 else 0.01*animation)

    ## show outcome
    average_click_rate = num_clicks/num_users
    best_click_rate = Theoretical.optimal_click_rate()
    print("%s"%"\n"*8)
    print(f"Strategy: {strategy.description()}")
    print(f"Number of users = {num_users}")
    print(f"Number of clicks = {num_clicks}")
    print(f"Click rate = {100*average_click_rate:1.2f}%")
    print(f"Theoretical best click rate = {100*best_click_rate:4.2f}%")

    ## plot the click rate & regret
    plt.figure(1)
    click_series = Empirical.get_click_rate_series()
    plt.plot(range(len(click_series)), click_series, '-')
    plt.xlabel("Number of ads offered")
    plt.ylabel("Click Rate")

    plt.figure(2)
    regret_series = Theoretical.get_regret_series()
    plt.plot(range(len(regret_series)), regret_series, '-')
    plt.xlabel("Number of ads offered")
    plt.ylabel("Regret")

    ## plot the arm selections
    plt.figure(3)
    arm_selection_series = Empirical.get_arm_selection_series()
    ad_type = Ad.AllArms.copy()
    ad_color = {0:"green",1:"blue",2:"pink",3:"yellow",4:"red"}
    for i in ad_color:
        plt.plot([],[],color=ad_color[i], label=ad_type[i], linewidth=5)
    plt.stackplot(range(len(arm_selection_series[ad_type[0]])),
                  arm_selection_series[ad_type[0]],
                  arm_selection_series[ad_type[1]], 
                  arm_selection_series[ad_type[2]], 
                  arm_selection_series[ad_type[3]], 
                  arm_selection_series[ad_type[4]], 
                  colors=list(ad_color.values()))
    plt.xlabel("Number of ads offered")
    plt.ylabel('Number shown') 
    plt.legend(loc="upper left") 
    plt.show()
    