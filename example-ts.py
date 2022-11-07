'''
**Digital advertising** is a form of marketing that targets online
users. A simple example of online marketing is where a website
embeds a small advertisement banner with the objective that 
users visiting the website will click the advertisement banner 
to explore the advertised products or services.

However, different users have different interests, and thus not
all types of advertisements will attract all users. We need to 
which type of advertisements is the most clicked by the users
and thus we can offer other users the same type of advertisements.

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
import numpy as np
from matplotlib.animation import FuncAnimation,PillowWriter

import matplotlib.pyplot as plt
from scipy.stats import beta

from mab import TS

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

    def expected_click_rate(arm) -> float:
        '''This is commonly notated as $\mu(a)$.'''
        return Ad.Type[arm]

    def optimal_click_rate() -> float: 
        '''This is commonly notated as $\mu^*$, which is
        $\max_{a\in A} \mu(a)$
        '''
        return max([mu_a for mu_a in list(Ad.Type.values())])

    def regret(t) -> float:
        '''This is commonly notated as $R(T)$, which is the regret
        at round $T$. It is calculated by 
        $R(T) = T \mu^* - \sum_{t=1}^T \mu(a_t)$
        where $a_t$ is the arm selection history.
        '''
        optimal = Theoretical.optimal_click_rate() * t  # optimal click rate
        experienced = 0                             # experienced click rate
        for arm in Ad.AllArms:
            experienced += Theoretical.expected_click_rate(arm) * Historical.get_arm_count(arm)
        regret_at_t = optimal - experienced
        Theoretical.regret_series.append(regret_at_t)
        return regret_at_t

    def get_regret_series():
        return Theoretical.regret_series


######################################################################
## Historical result keeper (static class)
######################################################################

class Historical:

    click_selections = [] # store the history of click selections
    click_outcomes = []   # store the history of click outcomes
    count_selection = {}  # store the total count of each arm selection
    alpha_series = {}
    beta_series = {}

    @staticmethod
    def report(arm,outcome):
        Historical.click_outcomes.append(outcome)
        Historical.click_selections.append(arm)
        if arm not in Historical.count_selection: # if new arm?
            Historical.count_selection[arm] = 0
        Historical.count_selection[arm] += 1
        for a in Ad.AllArms:
            if a not in Historical.alpha_series:   # if new arm?
                Historical.alpha_series[a] = [1]   # then initialize the lists
                Historical.beta_series[a] = [1]
            alpha = Historical.alpha_series[a][-1]
            beta = Historical.beta_series[a][-1]
            if a==arm:
                alpha += outcome
                beta  += 1-outcome
            Historical.alpha_series[a].append(alpha)
            Historical.beta_series[a].append(beta)

    @staticmethod
    def get_arm_count(arm):
        if arm not in Historical.count_selection:
            return 0
        return Historical.count_selection[arm]

    @staticmethod
    def get_click_rate():
        return sum(Historical.click_outcomes)/len(Historical.click_outcomes)

    @staticmethod
    def get_click_rate_series():
        click_rate_series = []
        click_rate_total = 0
        click_rate_size = 0
        for click in Historical.click_outcomes:
            click_rate_total += 1 if click else 0
            click_rate_size += 1
            click_rate_series.append(click_rate_total/click_rate_size)
        return click_rate_series

    @staticmethod
    def get_arm_selection_series():
        arm_selection_series = {}
        for arm in Ad.AllArms:
            arm_selection_series[arm] = [0]
        for selected_arm in Historical.click_selections:
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
        '''Will this client clicks this advert?'''
        click_prob = random.randint(0,99)
        if click_prob<100*Ad.Type[ad]:
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
    num_users = 2000 # number of users to visit the website
    num_clicks = 0   # number of clicks collected

    ## setup MAB (pick one)
    #mab = MAB()       # simple MAB agent
    #mab = UCB1_MAB()  # UCB MAB agent
    mab = TS()         # Thomspon Sampling

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

    ## print heading for the animation
    last_ucb = {}
    for ad_type in Ad.AllArms: last_ucb[ad_type] = 0
    print(f"Testing {mab.description()}\n")
    print(" Ad      Average  Last   Ad shown")
    print("type      reward  drawn  to users")
    print("-------------------------------")

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
        Historical.report(offered_ad, click_reward)
        mab.update_reward(arm=offered_ad, reward=click_reward)

        Theoretical.regret(round)

        ## show animation
        for arm in Ad.AllArms:
            r = mab.get_reward(arm)
            len_count_bar = int(50*Historical.get_arm_count(arm)/round)
            print(f"\033[K> {arm:8s} {r:5.2f}  ",end="")
            print(f"{mab.get_last_drawn_value(arm):3.2f}  ",end="")
            print("*" if arm==offered_ad else " ",end="")
            print("[%s] %d"%("="*len_count_bar,Historical.get_arm_count(arm)))
        current_click_rate = Historical.get_click_rate()
        current_regret = Theoretical.regret(round)
        print(f"\nClick rate = {current_click_rate:5.2f}")
        print(f"Regret = {current_regret:5.2f}")
        print("\033[9A")
        time.sleep(0.05)

    ## show outcome
    average_click_rate = num_clicks/num_users
    best_click_rate = Theoretical.optimal_click_rate()
    print("%s"%"\n"*8)
    print(f"Strategy: {strategy.description()}")
    print(f"Number of users = {num_users}")
    print(f"Number of clicks = {num_clicks}")
    print(f"Click rate = {100*average_click_rate:1.2f}%")
    print(f"Theoretical best click rate = {100*best_click_rate:4.2f}%")

    ## create animated beta distributions
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    ad_line_color = {"toys":"g-","cars":"b-","sports":"m-","holidays":"y-","foods":"r-"}
    def animate(i):
        i *= 5 # step
        ax.clear()
        ax.set_ylim([0,18])
        ax.text(0, 17, f"round = {i}")
        x = np.linspace(0, 1.0, 100)
        for arm in Ad.AllArms:
            y = beta.pdf(x, Historical.alpha_series[arm][i], Historical.beta_series[arm][i])
            ax.plot(x, y, ad_line_color[arm], label=arm+f" ({Ad.Type[arm]:3.2f})")
        ax.legend(loc="upper right") 
    ani = FuncAnimation(fig, animate, frames=150, interval=20, repeat=False)
    plt.close()
    ani.save("beta_distributions.gif", dpi=300, writer=PillowWriter(fps=10))
