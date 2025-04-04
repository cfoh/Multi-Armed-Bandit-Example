'''
Thompson Sampling Technique. 
'''

from math import ceil
import random
import time
import numpy as np
from matplotlib.animation import FuncAnimation,PillowWriter

import matplotlib.pyplot as plt
from scipy.stats import beta

from mab import TS
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
    alpha_series = {}
    beta_series = {}

    @staticmethod
    def report(arm,outcome):
        Empirical.click_outcomes.append(outcome)
        Empirical.click_selections.append(arm)
        if arm not in Empirical.count_selection: # if new arm?
            Empirical.count_selection[arm] = 0
        Empirical.count_selection[arm] += 1
        for a in Ad.AllArms:
            if a not in Empirical.alpha_series:   # if new arm?
                Empirical.alpha_series[a] = [1]   # then initialize the lists
                Empirical.beta_series[a] = [1]
            alpha = Empirical.alpha_series[a][-1]
            beta = Empirical.beta_series[a][-1]
            if a==arm:
                alpha += outcome
                beta  += 1-outcome
            Empirical.alpha_series[a].append(alpha)
            Empirical.beta_series[a].append(beta)

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
    animation   = True # True/False
    create_beta = False # to create animated beta function plot?

    ## setup MAB (pick one)
    mab = TS()         # Thompson Sampling

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
        Empirical.report(offered_ad, click_reward)
        mab.update_reward(arm=offered_ad, reward=click_reward)

        ## show animation
        for arm in Ad.AllArms:
            r = mab.get_reward(arm)
            len_count_bar = int(50*Empirical.get_arm_count(arm)/round)
            print(f"\033[K> {arm:8s} {r:5.2f}  ",end="")
            print(f"{mab.get_last_drawn_value(arm):3.2f}  ",end="")
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

    ## create animated beta distributions
    if create_beta:
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
                y = beta.pdf(x, Empirical.alpha_series[arm][i], Empirical.beta_series[arm][i])
                ax.plot(x, y, ad_line_color[arm], label=arm+f" ({Ad.Type[arm]:3.2f})")
            ax.legend(loc="upper right") 
        ani = FuncAnimation(fig, animate, frames=150, interval=20, repeat=False)
        plt.close()
        ani.save("beta_distributions.gif", dpi=300, writer=PillowWriter(fps=10))
