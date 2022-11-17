'''
Implementation of various Multi-Armed Bandit Algorithms and Strategies.
'''

import operator
import math
import random

######################################################################
## Simple Multi-Armed Bandit 
######################################################################
class MAB:
    '''
    Simple Multi-armed Bandit implementation.
    '''

    def __init__(self):
        '''Constructor.'''
        self.total_rewards = {}
        self.total_count = {}
        self.average_reward = {}

    def description(self) -> str:
        '''Return a string which describes the algorithm.'''
        return "Simple MAB"

    def update_reward(self, arm, reward):
        '''Use this method to update the algorithm which `arm` has been
        selected and what `reward` has been observed from the environment.'''
        if arm not in self.total_rewards: # new arm?
            self.total_rewards[arm] = 0
            self.total_count[arm] = 0
        self.total_count[arm] += 1
        self.total_rewards[arm] += reward
        self.average_reward[arm] = self.total_rewards[arm]/self.total_count[arm]

    def get_reward(self, arm):
        '''Get the reward for a particular `arm`.'''
        if arm not in self.average_reward: return 0
        return self.average_reward[arm]

    def get_arm_count(self, arm):
        '''Return how many times have this `arm` been selected.'''
        if arm not in self.total_count: return 0
        return self.total_count[arm]

    def get_best_arm(self):
        '''Return a tuple (arm,reward) representing the best arm and
        the corresponding average reward. If this arm has not been 
        seen by the algorithm, it simply returns (None,None).'''
        if len(self.average_reward)==0: 
            return (None,None)
        return max(self.average_reward.items(), key=operator.itemgetter(1))
 

######################################################################
## Upper Confidence Bound (UCB)
######################################################################
class UCB1(MAB):
    '''
    Upper Confidence Bound (UCB) implementation.
    '''

    def __init__(self, beta=1.0):
        '''Constructor.'''
        super().__init__()
        self.beta = beta
        self.overall_total_count = 0
        self.ucb = 0

    def description(self):
        '''Return a string which describes the algorithm.'''
        return "UCB MAB"

    def update_reward(self, arm, reward):
        '''Use this method to update the algorithm which `arm` has been
        selected and what `reward` has been observed from the environment.'''
        if arm not in self.total_rewards: 
            self.total_rewards[arm] = 0
            self.total_count[arm] = 0
        self.total_count[arm] += 1
        self.overall_total_count += 1
        self.ucb =  math.sqrt(2*self.beta*math.log(self.total_count[arm])/self.total_count[arm])
        ucb_reward = reward + self.ucb
        self.total_rewards[arm] += ucb_reward
        self.average_reward[arm] = self.total_rewards[arm]/self.total_count[arm]

    def get_last_ucb(self):
        return self.ucb


######################################################################
## Thompson Sampling Technique
######################################################################
class TS:
    '''
    Multi-armed Bandit with Thompson Sampling technique.
    '''

    def __init__(self):
        '''Constructor.'''
        self.total_count = {}
        self.alpha = {}
        self.beta = {}
        self.last_drawn = {}

    def description(self) -> str:
        '''Return a string which describes the algorithm.'''
        return "Multi-armed Bandit with Thompson Sampling technique"

    def update_reward(self, arm, reward):
        '''Use this method to update the algorithm which `arm` has been
        selected and what `reward` (must be either 0 or 1) has been observed 
        from the environment.'''
        if arm not in self.total_count: # new arm?
            self.alpha[arm] = 1
            self.beta[arm] = 1
            self.total_count[arm] = 0
            self.last_drawn[arm] = 0
        self.total_count[arm] += 1
        self.alpha[arm] += reward
        self.beta[arm]  += 1-reward

    def get_reward(self, arm):
        '''Get the reward for a particular `arm`. 
        This is $\frac{\alpha-1}{(\alpha-1)+(\beta-1)}$.'''
        if arm not in self.total_count: return 0
        return (self.alpha[arm]-1) / (self.alpha[arm]-1+self.beta[arm]-1)

    def get_arm_count(self, arm):
        '''Return how many times have this `arm` been selected.'''
        if arm not in self.total_count: return 0
        return self.total_count[arm]

    def get_best_arm(self):
        '''Return a tuple (arm,reward) representing the best arm and
        the corresponding average reward. If this arm has not been 
        seen by the algorithm, it simply returns (None,None).'''
        best_arm = { "arm":None, "value":0.0 }
        for arm in self.total_count:
            self.last_drawn[arm] = random.betavariate(self.alpha[arm],self.beta[arm])
            if self.last_drawn[arm]>=best_arm["value"]:
                best_arm["arm"] = arm
                best_arm["value"] = self.last_drawn[arm]
        if best_arm["arm"] is None: 
            return (None,None)
        return (best_arm["arm"],best_arm["value"])

    def get_last_drawn_value(self, arm):
        if arm not in self.last_drawn: return 0
        return self.last_drawn[arm]


######################################################################
## Boltzmann Exploration (Softmax)
######################################################################
class SoftMax(MAB):
    '''
    Boltzmann Exploration (Softmax).
    '''

    def __init__(self, tau=1.0):
        '''Constructor.'''
        super().__init__()
        self.tau = tau

    def description(self) -> str:
        '''Return a string which describes the algorithm.'''
        return "Boltzmann Exploration (Softmax)"

    def get_best_arm(self):
        '''Return a tuple (arm,reward) representing the best arm and
        the corresponding average reward. If this arm has not been 
        seen by the algorithm, it simply returns (None,None).'''
        if len(self.average_reward)==0: 
            return (None,None) # nothing in Q-table yet, do exploration
        arm_list = [arm for arm in self.average_reward]
        arm_weight = [math.exp(reward/self.tau) for reward in self.average_reward.values()]
        # note that we don't need to divide the denominator because 
        # `random.choices()` will scale `arm_weight` automatically
        choice = random.choices(arm_list,arm_weight)[0]
        return (choice,self.average_reward[choice])

    def get_prob_list(self):
        '''Get the probability dictionary for all arms. Each quantity describes
        the probability that an arm will be picked.'''
        arm_prob = {}
        weigth_sum = 0
        if len(self.average_reward)!=0:
            for arm,reward in self.average_reward.items():
                arm_prob[arm] = math.exp(reward/self.tau)
                weigth_sum += arm_prob[arm]
            for arm in arm_prob:
                arm_prob[arm] /= weigth_sum
        return(arm_prob)

######################################################################
## Simple Discrete Contextual MAB
## Using Multi-UCB1
######################################################################
class CMAB:
    '''
    Simple Discrete Contextual Multi-armed Bandit implementation
    using Multi-UCB1.
    '''

    def __init__(self):
        '''Constructor.'''
        self.mab = {}

    def description(self):
        '''Return a string which describes the algorithm.'''
        return "Contextual MAB using Multi-UCB1"

    def update_reward(self, arm, reward, context=None):
        '''Use this method to update the algorithm which `arm` has been
        selected under which `context, and what `reward` has been observed 
        from the environment.'''
        if context not in self.mab: 
            self.mab[context] = UCB1() # we use UCB1 model for each context
        self.mab[context].update_reward(arm, reward)

    def get_reward(self, arm, context=None):
        '''Get the reward for a particular `arm` under this `context`.'''
        if context not in self.mab: # new context?
            return 0 
        return self.mab[context].get_reward(arm)

    def get_best_arm(self, context=None):
        '''Return a tuple (arm,reward) representing the best arm and
        the corresponding average reward. If this context has not been 
        seen by the algorithm, it simply returns (None,None).'''
        if context not in self.mab: return (None,None)
        return self.mab[context].get_best_arm()


######################################################################
## Simple Discrete Contextual MAB
## Using Summarized Contexts
######################################################################
class CMAB2(MAB):
    '''
    Simple Discrete Contextual Multi-armed Bandit implementation
    using Summarized Contexts. This class extends simple MAB,
    extending other models are also possible, e.g. UCB1.
    '''

    def __init__(self):
        '''Constructor.'''
        super().__init__()

    def description(self):
        '''Return a string which describes the algorithm.'''
        return "Contextual MAB using Summarized Contexts"

    def context(self, feature, action=None):
        '''Return the context summarizing feature and action.'''
        return (feature,action)

    def update_reward(self, context, reward):
        '''Use this method to update the algorithm which `context` has been
        observed and what `reward` has been obtained from the environment.'''
        super().update_reward(context,reward)

    def get_best_arm(self, context):
        '''Return a tuple (action,reward) representing the best arm and
        the corresponding average reward. If this context has not been 
        seen by the algorithm, it simply returns (None,None).'''
        best_action = (None,None)  # (action,reward)
        for cnx in self.average_reward:
            if cnx[0]==context[0]: # context=(feature,action)
                if best_action[0] is None or best_action[1]<self.average_reward[cnx]:
                    best_action = (cnx[1],self.average_reward[cnx])
        return best_action


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

class EpsilonDecreasing(BaseStrategy):
    def __init__(self,exponent):
        self.exponent = exponent
    def description(self):
        return f"Epsilon-Decreasing, exponent = {self.exponent}"
    def is_exploration(self,round):
        epsilon = round**(self.exponent)
        return random.random()<epsilon

class ExplorationFirst(BaseStrategy):
    def __init__(self,switch_round):
        self.switch_round = int(switch_round)
    def description(self):
        return f"Explore-First for {self.switch_round} rounds"
    def is_exploration(self,round):
        return round<self.switch_round

