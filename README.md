# Multi-Armed Bandit (MAB)

<table>
<tr><td colspan="2"><b>
Chapter 1: Simple MAB
</b></td></tr>
<tr>
<td valign="top">
    <ul>
        <li><a href=#intro>Introduction</a></li>
        <li><a href=#outcomes>The Outcomes</a></li>
        <li><a href=#results>Result Plotting</a></li>
        <li><a href=#codes>MAB/CMAB Implementation</a></li>
    </ul>
</td>
<td>
  Demo:<br>
  <img src="https://user-images.githubusercontent.com/51439829/197750117-97e50e9b-7fc0-4a16-ab77-fb1ddc8434c4.gif" width="400">
</td>
<tr><td colspan="2">
The above demo shows how the ML agent offers advertisement to `Age Group 2`. Users in this group prefer `sports`, with 40% chance to click. The ML agent will slowly discover this by noticing higher rewards from `sports` and lower from others, and will present mostly `sports` to the users (i.e. the bar for `sports` counting the number of advertisements shown to users gets longer over the time).
<br>
When the reward for `sports` is much greater than others, the agent will certainly show `sports` to the users. It may still show other ad types, but only during occasional exploration.
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2"><b>
Chapter 2: MAB with Upper Confidence Bound (UCB)
</b></td></tr>
</table>

## Introduction<a name=intro></a>

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
It shows the likelihood of users in each age group clicking 
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

With Contextual MAB, our setup is:
- the `context` is the age group
- the `arms` are advertisement type to offer
- the `reward` is 100 if a click is registered, otherwise 0

## Outcomes<a name=outcomes></a>

We measure the effectiveness of our strategy using `click rate`. It is the percentage that a user will click and explore the offered advertisement. By setting epsilon to 0.15, we achieve around 48% of click rate. With the user behaviour, the theoretical best click rate we can achieve is 56%. As you can see, we are not far from the theoretical best.

```
Epsilon = 0.15
Number of users = 10000
Number of clicks = 5060
Click rate = 50.60%
Theoretical best click rate = 56.00%

Age group   Ad to offer    Expected CMAB reward
===============================================
    0         toys              80.37
    1         cars              47.20
    2         sports            40.16
    3         holidays          49.48
    4         foods             58.16
```

The theoretical result is calculated based on the assumption that we know the user behaviour matrix. As a result, knowing the user age group, we will always offer the most attractive advertisement to the user. For example, for Age Group 0 (or <25), we'll always show advertisement related to "Toys" as we know that 80% of the users will click and explore the advertisement, and thus our success rate is 80% too. So, the overall successful click rate (in %) is the average over all user age groups when we offered the most attractive advertisement, and this gives

<p align="center">
$\frac{80+50+40+50+60}{5} = 56$.
</p>

## Plots<a name=results></a>

We can also see how click rate evolves over the time for each user age group.

<img src="https://user-images.githubusercontent.com/51439829/197748313-5b9ea8d5-c44e-4f08-8173-0702415d8465.png" width="400">

See below, if we set `epsilon = 1.0`, we essentially force the ML agent to operate in 100% exploration. During exploration, the ML agent simply randomly picks a type of advertisements. The click rate for each age group is simply the average of the 5 click probabilities in the age group. So we have:

| Age Group:      | 0  | 1  | 2  | 3  | 4  |
|-----------------|----|----|----|----|----|
| Click rate (%): | 22 | 28 | 28 | 28 | 31 |

<img src="https://user-images.githubusercontent.com/51439829/197748387-ac5db93b-f6e9-41d4-bda1-395926ecf741.png" width="406">

## MAB & CMAB Implementation<a name=codes></a>

The most basic version of MAB and CMAB is very easy to implemenet. The value of an action (or arm) is the average reward which is calculated by

$Q_{k}(a) = \frac{1}{k}\left(r_1 + r_2 + \cdots + r_k\right)$

where $k$ is the number of times that action $a$ has been chosen in the past, $r_i$ is the $i$-th reward when action $a$ was chosen, and the sum $(r_1 + r_2 + \cdots + r_k)$ is the total reward. We can also compute $Q_{k}(a)$ recursively by

$Q_{k}(a) = \frac{1}{k} \left((k-1)\cdot Q_{k-1}(a) + r_k\right)$.

For MAB:
```Python
import operator

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
```

For CMAB based on the above MAB class:
```Python
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
```
