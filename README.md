# Multi-Armed Bandit (MAB)

<table>
<tr><td colspan="2"><b>
Chapter 1: Simple MAB
</b></td></tr>
<tr>
<td valign="top">
    <ul>
        <li><a href=#intro>Introduction</a></li>
        <li><a href=#codes>Implementation</a></li>
        <li><a href=#outcomes>The Outcomes</a></li>
        <li><a href=#results>Result Plotting</a></li>
        <li><a href=#next>What's Next?</a></li>
    </ul>
</td>
<td>
  Demo:<br>
  <img src="https://user-images.githubusercontent.com/51439829/197750117-97e50e9b-7fc0-4a16-ab77-fb1ddc8434c4.gif" width="400">
</td>
<tr><td colspan="2">
The above demo shows how the ML agent offers advertisements. Users prefer `sports`, with 40% chance to click. The ML agent initially exploited `cars` as it was receiving good click rate. As the agent explored more on `sports`, its click rate improved and became the best. The agent then continued to exploit `sports` with occasion exploration to other ads..
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2"><b>
More:<br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb">Chapter 2: Upper Confidence Bound (UCB) Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 3: Contextual Multi Armed Bandit</a></li>
</ul>
</b></td></tr>
</table>

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

The concept of MAB is very simple. In this example, the agent has 5 ads (or `arms`) to show to users. It switches between exploration and exploitation. 
- During exploration, it randomly picks an ad. After showing the ad, it observes whether the ad banner has been clicked by the user. The outcome is translated to `rewards` and recorded in a `reward table`.
- During exploitation, it checks the `reward table` to look for the ad that has the highest average reward. It then picks that ad to show to the user.

The average reward of an arm can be calculated directly by

$$Q_{k}(a) = \frac{1}{k}\left(r_1 + r_2 + \cdots + r_k\right)$$

where $k$ is the number of times that arm $a$ has been chosen in the past, $r_i$ is the $i$-th reward when arm $a$ was chosen, and the sum $(r_1 + r_2 + \cdots + r_k)$ is the total reward. We can also compute $Q_{k}(a)$ recursively by

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

## Outcomes<a name=outcomes></a>

We measure the effectiveness of our strategy by `click rate`. It is the percentage that a user will click and explore the offered advertisement. By setting epsilon to 0.15, we achieve around 36% of click rate. If somehow the ML agent knows the user behaviour, it will, of course, always offer `sports` which has the highest click rate among all, and so the theoretical optimal click rate is 40%. Our ML agent achieves 36% which is actually not far from the optimal click rate.

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

Another measure of the ML performance is `Regret`. It measures the reward gap between the picked arm and the best arm. Obviously, we want the gap to be small, i.e. the regret to be low. Let $a_t$ be the arm selected at round $t$ and the reward collected by selecting it is $\mu(a_t)$. Let $\mu^{\star}$ be the optimal average reward. Then the gap between the collected and optimal rewards is simply $\mu^{\star}-\mu(a_t)$. After accumulating $T$ rounds of regrets, we get

$$R(T) = \sum_{t=1}^{T} (\mu^{\star} - \mu(a_t)) = T \mu^{\star} - \sum_{t=1}^{T} \mu(a_t).$$

Imagine the ML agent made the following picks over 5 rounds, knowing $\mu^{\star}=0.4$, then $R(T)$ is:

| $T$                    |    1     |      2     |     3   |     4    |    5   |
|------------------------|----------|------------|---------|----------|--------|
| $a_t$                  | `sports` | `holidays` | `foods` | `sports` | `cars` |
| $\mu(a_t)$             |    0.4   |    0.35    |  0.25   |   0.4    |  0.3   |
| $\mu^{\star}-\mu(a_t)$ |    0     |    0.05    |  0.15   |   0      |  0.1   |
| $R(T)$                 |    0     |    0.05    |  0.20   |   0      |  0.30  |

## Plots<a name=results></a>

Here we plot the click rate evolves over the time below.

![mab-click-rate](https://user-images.githubusercontent.com/51439829/200084339-512843fa-633d-46c6-949e-07b9cc3b2fce.png)

In the following, we plot the regret evolution. As can be seen, the regret was climbing fast initially as `cars` was lucky to have higher click rate (or average reward) causing the agent to exploit it. After discovering that `sports` had a better average reward, it then started to exploit `sports`. Ideally the regret should stay flat after discovering `sports` being the best ad. But since the agent still occasionally explored other non-optimal arms, the regret continued to increase due to this.

![mab-regret](https://user-images.githubusercontent.com/51439829/200084369-acfbfea0-34f4-43b0-95de-58e4a605af49.png)

We can see which arms our ML agent picked over the time. As shown below, it indeed picked `cars` (in blue) at the beginning. After `sports` (in pink) took over to offer a better average reward, the agent then switched to exploit `sports` until the end of the simulation run.

![mab-arm-selection](https://user-images.githubusercontent.com/51439829/200084382-6bde7ef8-4da9-4cf3-888d-f98331e69f9a.png)

## What's Next?<a name=next></a>

While being simple, the algorithm runs a risk of making premature decision without collecting enough statistic from the environment. 

Take a look at the following demo. In this run, the agent had an unlucky start with `sports` as users didn't click the ad. As a result, the agent exploited `holidays`. While `sports` was explored, the occasional exploration wasn't enough to make up for the unlucky start. The potential of `sports` was unrevealed.

<img src="https://user-images.githubusercontent.com/51439829/200084414-6511572e-d61b-47c6-b7fc-a1a7a85b2783.gif" width="400">

Upper Confidence Bound (UCB) addresses the issue by giving extra reward to the arms that are less explored. We shall discuss this in the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb).
