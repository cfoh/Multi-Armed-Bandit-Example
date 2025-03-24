# Multi-Armed Bandit (MAB) ![Machine Learning: Multi-Armed Bandit](https://img.shields.io/badge/Machine%20Learning-Multi--Armed%20Bandit-blueviolet) ![MIT License](https://img.shields.io/badge/License-MIT-green?logo=github)

<table>
<tr><td colspan="2"><b>
Chapter 1: Simple MAB
</b></td></tr>
<tr>
<td valign="top">
    <ul>
        <li><a href=#intro>Introduction</a></li>
        <li><a href=#codes>Implementation</a></li>
        <li><a href=#strategy>Exploration-Exploitation</a></li>
        <li><a href=#outcomes>Outcomes</a></li>
        <li><a href=#results>Result Plotting</a></li>
        <li><a href=#next>What's Next?</a></li>
    </ul>
</td>
<td>
  Demo:<br>
  <img src="https://user-images.githubusercontent.com/51439829/200315041-54cde21e-16c1-4350-8973-a6844ef21534.gif" width="400">
  <br><b>NOTE for Windows User:</b> Animation will only work in <a href=https://devblogs.microsoft.com/commandline/introducing-windows-terminal>Windows Terminal</a> (available in Windows 10 and later). Don't use `PowerShell` and `Command Prompt`, as they do not support Escape Codes.</b>
</td>
<tr><td colspan="2">
The above demo shows how the ML agent offers advertisements. Users prefer `sports`, with 40% chance to click. The ML agent initially exploited `cars` as it was receiving good click through rate. As the agent explored more on `sports`, its click through rate improved and became the best. The agent then continued to exploit `sports` with occasion exploration to other ads.
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2">
<b>Contents</b><br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example">Chapter 1: Simple Multi Armed Bandit</a>&nbsp;<img src="https://img.shields.io/badge/This-Chapter-blue"></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb">Chapter 2: Upper Confidence Bound (UCB) Algorithm</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax">Chapter 3: Boltzmann Exploration (Softmax)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts">Chapter 4: Thompson Sampling Technique</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 5: Contextual Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2">Chapter 6: Contextual Multi Armed Bandit (more)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn">Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)</a></li>
<li><b>Recommended reading</b>: 
    <ul>
    <li><a href="https://arxiv.org/pdf/1904.07272"><img src="https://img.shields.io/badge/PDF-F40F02"> Introduction to Multi-Armed Bandits by Aleksandrs Slivkins (Microsoft Research NYC)</a></li>
    </ul>
</ul>
</td></tr>
</table>

![slot-machines](https://user-images.githubusercontent.com/51439829/215224983-ad95a546-49a4-4cbf-b559-36b558077675.png)

## Introduction<a name=intro></a>

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

## MAB Implementation<a name=codes></a>

Here we're dealing with **Stochastic Bandit** problem. In other words, the rewards we observe from the environment follow a particular random behaviour which is unknown to us. We want to develop a policy that we can learn the rewards over the time and improve our decision.

The process of MAB is very simple. In this example, the agent has 5 ads (or `arms`) to show to users. It switches between exploration and exploitation. 
- During exploration, it randomly picks an ad. After showing the ad, it observes whether the ad banner has been clicked by the user. The outcome is translated to `rewards` and recorded in a `reward table`.
- During exploitation, it checks the `reward table` to look for the ad that has the highest average reward. It then picks that ad to show to the user.

The average reward of an arm can be calculated directly by:

$$Q_{k}(a) = \frac{1}{k}\left(r_1 + r_2 + \cdots + r_k\right)$$

where $k$ is the number of times that arm $a$ has been chosen in the past, $r_i$ is the $i$-th reward when arm $a$ was chosen, and the sum $(r_1 + r_2 + \cdots + r_k)$ is the total reward. We can also compute $Q_{k}(a)$ recursively by:

$$Q_{k}(a) = \frac{1}{k} \left((k-1)\cdot Q_{k-1}(a) + r_k\right).$$

The implementation using the first direct method:

```Python
class MAB:

    def __init__(self):
        '''Constructor.'''
        self.total_rewards = {}
        self.total_count = {}
        self.average_reward = {}

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
```

## Exploration-Exploitation Strategy<a name=strategy></a>

In the previous section, we see that the ML agent switches between exploration and exploitation. Exploration can help the agent to establish new finding, but the finding might not produce good outcomes. On the other hand, Exploitation ensures the agent picks the best option based on what have already discovered, but might get stuck at a local maximal if the discovery is insufficient. This is called **exploration-exploitation dilemma**.

There are two popular strategies:
- **Explore-First**: We explore random arms for some rounds, 
  then we switch to exploitation for the remaining rounds of operation.
- **Epsilon-Greedy**: For every round, we explore a random arm with $\epsilon$
  probability. We can set $\epsilon$ to a fixed value or use a function. We often use an $\epsilon$-decreasing function, e.g. $\epsilon=t^{-\frac{1}{3}}$ where $t$ is the current round so that the agent performs more exploitations as time progresses.

The selection of strategy is critical if the time horizon is finite. That is, you only have a finite number of rounds to explore and exploit. For example, you are visiting a new town for a week, should you explore a new restaurant for every dinner or exploit a good one when you discovered one? This specific problem has been studied and the result is quite interesting, see [Gittins Index](https://en.wikipedia.org/wiki/Gittins_index).

## Outcomes<a name=outcomes></a>

> **Note**: Animation issue in Windows `Command Prompt` or `PowerShell`? Use `Windows Terminal` which supports escape codes.

We measure the effectiveness of our strategy by `click through rate` (CTR) or simply called `click rate`. It is the percentage that a user will click and explore the offered advertisement. We use Epsilon-Greedy strategy with the setting $\epsilon=0.15$, and we achieve around 36% of click rate. If somehow the ML agent knows the user behaviour, it will, of course, always offer `sports` which has the highest click rate among all, and so the theoretical optimal click rate is 40%. Our ML agent achieves 36% which is actually not far from the optimal click rate.

```console
Testing Simple MAB

Ad_type   Reward  Ad_shown_to_users
-----------------------------------
> toys      0.07  [=] 66
> cars      0.30  [=========] 393
> sports    0.40 *[===================================] 1402
> holidays  0.33  [=] 74
> foods     0.21  [=] 60

Click rate =  0.36
Regret = 73.80

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 2000
Number of clicks = 719
Click rate = 35.95%
Theoretical best click rate = 40.00%
```

The top part of the printout shows the average reward recorded in the ML agent for each ad, and the number of times that the ad is shown to users. As can be seen, the agent correctly recorded `sports` being the highest reward among all.

Another measure of the ML performance is `Regret`. It measures the reward gap between the picked arm and the best arm. Obviously, we want the gap to be small, i.e. the regret to be low. Let $a_t$ be the arm selected at round $t$ and the expected reward collected by selecting it is $\mu(a_t)$. Let $\mu^{\star}$ be the optimal average reward. Then the gap between the collected and optimal rewards is simply $\mu^{\star}-\mu(a_t)$. After accumulating $T$ rounds of regrets, we get:

$$R(T) = \sum_{t=1}^{T} (\mu^{\star} - \mu(a_t)) = T \mu^{\star} - \sum_{t=1}^{T} \mu(a_t).$$

Imagine the ML agent made the following picks over 5 rounds, knowing $\mu^{\star}=0.4$, then $R(T)$ for this particular experiment run is:

| $T$                    |    1     |      2     |     3   |     4    |    5   |
|------------------------|----------|------------|---------|----------|--------|
| $a_t$                  | `sports` | `holidays` | `foods` | `sports` | `cars` |
| $\mu(a_t)$             |    0.4   |    0.35    |  0.25   |   0.4    |  0.3   |
| $\mu^{\star}-\mu(a_t)$ |    0     |    0.05    |  0.15   |   0      |  0.1   |
| $R(T)$                 |    0     |    0.05    |  0.20   |   0.20   |  0.30  |

Note that $R(T)$ is a random process. We're more interested in $E[R(T)]$, the mean of $R(T)$. Thus we need run the experiment many times to obtain $E[R(T)]$.

## Plots<a name=results></a>

Here we plot the click rate evolving over the time below.

![mab-click-rate](https://user-images.githubusercontent.com/51439829/200084339-512843fa-633d-46c6-949e-07b9cc3b2fce.png)

In the following, we plot the regret evolution for one experiment run. As can be seen, the regret was climbing fast initially as `cars` was lucky to have higher click rate (or average reward) causing the agent to exploit it. After discovering that `sports` had a better average reward, it then started to exploit `sports`. Ideally the regret should stay flat after discovering `sports` being the best ad. But since we use Epsilon-Greedy strategy, the agent still occasionally explored other non-optimal arms, the regret continued to increase linearly with a slower rate due to this.

![mab-regret](https://user-images.githubusercontent.com/51439829/200084369-acfbfea0-34f4-43b0-95de-58e4a605af49.png)

We can see which arms our ML agent picked over the time. As shown below, it indeed picked `cars` (in blue) at the beginning. After `sports` (in pink) took over to offer a better average reward, the agent then switched to exploit `sports` until the end of the simulation run.

![mab-arm-selection](https://user-images.githubusercontent.com/51439829/200084382-6bde7ef8-4da9-4cf3-888d-f98331e69f9a.png)

## What's Next?<a name=next></a>

While being simple, the algorithm runs a risk of making premature decision without collecting enough statistic from the environment. 

Take a look at the following demo. In this run, the agent had an unlucky start with `sports` as users didn't click the ad. As a result, the agent exploited `holidays`. While `sports` was explored, the occasional exploration wasn't enough to make up for the unlucky start. The potential of `sports` was unrevealed.

<img src="https://user-images.githubusercontent.com/51439829/200084414-6511572e-d61b-47c6-b7fc-a1a7a85b2783.gif" width="400">

Upper Confidence Bound (UCB) addresses the issue by giving extra reward to the arms that are less explored. We shall discuss this in the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb).
