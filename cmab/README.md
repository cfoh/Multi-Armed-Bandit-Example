# Chapter 5: Contextual Multi-Armed Bandit

<table>
<tr><td colspan="2"><b>
Chapter 5: Contextual Multi Armed Bandit
</b></td></tr>
<tr>
<td valign="top">
    <ul>
        <li><a href=#intro>Introduction</a></li>
        <li><a href=#codes>Implementation</a></li>
        <li><a href=#outcomes>The Outcomes</a></li>
        <li><a href=#next>What's Next?</a></li>
    </ul>
</td>
<td>
  Demo:<br>
  <img src="https://user-images.githubusercontent.com/51439829/200530420-104d1d82-1178-46b4-a033-e4e3e8f2d896.gif" width="400">
</td>
<tr><td colspan="2">

The above demo compares the ML performance when the model uses or doesn't use contexts. MAB that doesn't use contexts struggles to pick the best ad to show to users as the user behaviours are different for different user groups (or contexts). CMAB uses context information to differentiate user groups. Once learned, it is more likely to pick the best ad than the MAB.
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2">
<b>Contents</b><br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example">Chapter 1: Simple Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb">Chapter 2: Upper Confidence Bound (UCB) Algorithm</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax">Chapter 3: Boltzmann Exploration (Softmax)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts">Chapter 4: Thompson Sampling Technique</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 5: Contextual Multi Armed Bandit</a>&nbsp;<img src="https://img.shields.io/badge/This-Chapter-blue"></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2">Chapter 6: Contextual Multi Armed Bandit (more)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn">Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)</a></li>
</ul>
</td></tr>
</table>

## Introduction<a name=intro></a>

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

## Implementation<a name=codes></a>

With a small set of discrete and independent contexts, considering applying UCB1 based our base method, we can simply use a separate instance of UCB1 for each context. In other words, each UCB1 instance independently learns and optimizes its decision making for its own context. This approach is described by Li *et al.* as `UCB (seg)` in their work [here](https://arxiv.org/pdf/1003.0146.pdf).

The following is the implementation:

```python
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
```

## Outcomes<a name=outcomes></a>

In our test, we run and compare the performance of two models below:
- A simple MAB that ignores context
- The CMAB model described above

We expect that since the simple MAB doesn't take the contexts into account, its learning is unable to converge to the optimal arm as when the context changes, the user behaviour changes, so is the optimal arm. On the other hand, our CMAB model uses a dedicated UCB1 to serve each user group with the same behaviour, and hence each UCB1 is able to discover the optimal arm for each user group.

The following compares the number of optimal arms played by each model. This is also known as the `no regret` option. Ideally, we want the no regret cases to be as high as possible. Comparing the two models after 10000 visitors, we see that about 2/3 of them are shown with the best ad type when using our CMAB model. This quantity is reduced down to just over 1/5 if we ignore contexts.

```console
Number of visitors = 10000
Optimal arm played:
> MAB  [============] 2059
> CMAB [========================================] 6759

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 10000
Theoretical best click rate = 56.0%

                     MAB   CMAB 
                   -------------
Number of clicks =  3061   4828
Click rate       = 30.6%  48.3%
```

We also plot the no regret evolution over the course of the experiment below. We see the immediate performance gain right from the beginning. As the learning processes, CMAB widens the gap further to produce about 3 times more `no regret` options than that of MAB.

<img src="https://user-images.githubusercontent.com/51439829/200530576-6dc7bdad-f373-4ffa-90bb-5d7da4098f73.png" width="400"/>

## What's Next?<a name=next></a>

Contextual Multi Armed Bandit can also be implemented differently. We can summarize user features and actions to form contexts. With all information summarized in the context, we can then use a single ML agent to manage all contexts. We shall discuss this implementation in the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2).
