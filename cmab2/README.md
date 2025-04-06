# Chapter 6: Contextual Multi-Armed Bandit (more)

<table>
<tr><td colspan="2"><b>
Chapter 6: Contextual Multi Armed Bandit (more)
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
  <img src="https://user-images.githubusercontent.com/51439829/202530268-e0a7aad7-c8b8-46e2-af34-a9b7492a3615.gif" width="400">
</td>
<tr><td colspan="2">
The above is a simple demo showing how a single ML agent manage contexts. We have 5 user age groups and 5 types of advertisements to offer to each user. The demo shows the number of ads shown for each age group. As can be seen, the ML agent can differentiate user age groups and serve them with different types of ads that maximizes click through rate.
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
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 5: Contextual Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2">Chapter 6: Contextual Multi Armed Bandit (more)</a>&nbsp;<img src="https://img.shields.io/badge/This-Chapter-blue"></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn">Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/linucb">Chapter 8: Linear Bandit and LinUCB</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn2">Chapter 9: Deep Learning Based Bandit for C-MAB</a></li>
</ul>
</td></tr>
</table>

## Introduction<a name=intro></a>

In the chapter, we introduce Contextual Multi Armed Bandit. In our example, we use different user age groups as the context to partition the learning. Precisely, we create 5 independent ML agents, each looks after a specific user age group.

In the [work](https://arxiv.org/pdf/1003.0146.pdf) by Li *et al.*, user features and actions are summarized into context. In this chapter, we shall implement MAB using this approach. With this implementation, we only have a single ML agent. Because the contexts contain both user features and actions, it is possible for the ML agent to focus on a specific user feature and learn the best action for that user feature.

## Implementation<a name=codes></a>

In the implementation, we summarize user age groups and ad types into contexts, and use the context as the arm to update rewards. When deciding on the best arm, considering picking the arm with the highest average reward, we focus on those that contains the same user age group as observed from the environment, pick the context with the highest reward and execute the corresponding action.

With this implementation, the agent has 25 contexts. However, for each decision making, the agent only select 5 contexts that contain the same user age group to find the highest average reward. This is equivalent to partitioning user age groups. The following is the implementation:

```python
class CMAB2(MAB):
    '''
    Simple Discrete Contextual Multi-armed Bandit implementation
    using Summarized Contexts. This class extends simple MAB,
    extending other models are also possible, e.g. UCB1.
    '''

    def __init__(self):
        '''Constructor.'''
        super().__init__()

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
```

## Outcomes<a name=outcomes></a>

To confirm that the ML agent can differentiate user age group and picks the optimal arm for each user age group, we show the table illustrating which type of ads (i.e. action) the agent presents to each user age group (i.e. context). Recall the environment:

```
  The Environment
+-------------------+--------------------------------------+
| Ad Type           |         Age group (context)          |
| (action)          |  <25    26-35   36-45   46-55  >55   |
+-------------------+--------------------------------------+
| Toys & Games      | [80%]    15%     10%     5%     5%   |
| Cars              |   5%    [50%]    30%    15%    10%   |
| Sports            |  15%     30%    [40%]   30%    30%   |
| Holiday Packages  |   5%     20%     35%   [50%]   50%   |
| Foods & Health    |   5%     25%     25%    40%   [60%]  |
+-------------------+--------------------------------------+
```

The aim of the C-MAB algorithm is to observe the feedback from the environment to establish a context-action table that closely approximates the ground truth from the environment. 

In the above, those with brackets are the best among others within the same age group. Based on that, we expect `toys`, `cars`, `sports`, `holidays` and `foods` to be the optimal choice for age groups 1, 2, 3, 4 and 5 respectively. Indeed, the agent eventually exploits those optimal choices for the age groups as shown below.

```console
Testing Contextual MAB using Summarized Contexts

Number of ads presented:
Age Group:      1     2     3     4     5
            ------------------------------
  toys       1781    72    69    71    72  
  cars         63  1706    55    63    60  
  sports       58    69  1690    56    80  
  holidays     58    59    94  1745    65  
  foods        61    81   121    53  1698  

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 10000
Theoretical best click rate = 56.0%

Number of clicks =  5054
Click rate       = 50.5%
```

## What's Next?<a name=next></a>

The MAB and C-MAB techniques so far we have studied use tables to store information about the data seen. Instead of using a table, we can also use a neural network. The benefit of using a neural network is its capability to structure data and infer unseen data. In the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn), we shall illustrate how to directly replace action-value table in C-MAB using a neural network. 
