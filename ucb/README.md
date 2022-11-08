# Upper Confidence Bound (UCB) Algorithm

<table>
<tr><td colspan="2"><b>
Chapter 2: Upper Confidence Bound
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
  <img src="https://user-images.githubusercontent.com/51439829/200187042-50ea8da6-3675-4d83-82af-e6e725785985.gif" width="400">
</td>
<tr><td colspan="2">
From the above demo, we can see that the ML agent gives preference to those arms explored less. This is because arms with fewer exploration gives higher UCB. As UCB is part of the reward, those arms will produce higher overall rewards.
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2">
<b>Back to:</b><br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example">Chapter 1: Multi Armed Bandit</a></li>
</ul>
<b>More:</b><br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts">Chapter 3: Thompson Sampling Technique</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 4: Contextual Multi Armed Bandit</a></li>
</ul>
</td></tr>
</table>

## Introduction<a name=intro></a>

In Chapter 1, we saw that the ML agent may overlook potential best arm due to unlucky start and prematurely conclude that the arm wasn't worth exploiting. How do we let the agent know some arms actually have good potential and should be exploited even the average reward at that time isn't impressive?

Luckily, we can apply Chernoff-Hoeffding bound to derive confidence interval. Essentially, we establish the probability that the next reward is bounded by some interval (also called radius) after seeing $N$ samples of rewards. If the probability is set very high, we say that we're very confident that the next reward is bounded by the interval, hence called confidence interval. While there are still a very small chance that the next reward may fall outside of the interval, we just think that this is so rare that we can ignore.

Back to the interval, it has an upper bound and a lower bound. We're interested in the upper bound, since this tells the potential of the next reward. Let $\bar{\mu}(a)$ be the empirical average reward of arm $a$ after exploring the arm $N$ times, and $\mu(a)$ be the true average reward of arm $a$. Skipping all detail derivations, we can show that:

$$Pr[|\bar{\mu}(a)-\mu(a)|\le r] \ge 1-\frac{2}{T^{2\alpha}}$$

where 

$$r = \sqrt{\frac{\alpha\beta\ln(T)}{N}}$$

and $r$ is the confidence interval radius. The above inequality says that we can expect the next reward to fall within the interval $[\bar{\mu}(a)-r,\bar{\mu}(a)+r]$ with the probability $1-\frac{2}{T^{2\alpha}}$.

With the above, the upper confidence bound of the next reward is thus:

$$\text{UCB}(a) = \bar{\mu}(a) + \sqrt{\frac{\alpha\beta\ln(T)}{N}}.$$

Having the UCB, instead of using the observed average reward $\bar{\mu}(a)$ to decide which arm should be picked, we shall now use the upper bound reward $\text{UCB}(a)$ which includes the observed reward $\bar{\mu}(a)$ and the confidence bound radius $r$. Note that applying the above result requires further [treatment and proof](https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Auer+al:2002.pdf), we will not discuss them here.

## Implementation<a name=codes></a>

The UCB system takes a few parameters to construct $\alpha$, $\beta$ and $T$. The classical UCB (or UCB1) has the following settings:
- $\alpha$: It controls the failure where a future reward escapes the bound.
  This value should be sufficiently small to ensure that the probability of 
  failure is very small. In many implemenations, we set $\alpha=2$.
- $\beta$: It is a scaler related to the reward range. 
  If rewards are within a range $[u,v]$, then $\beta = (v-u)^2$. For the 
  reward range of $[0,1]$, $\beta=1$.
- $T$: It is a fixed parameter. Ideally, this quantity should be large to ensure 
  low failure. We often set it to the number of rounds. 
  Consequently, the UCB radius also increases accordingly, more so for those insufficiently explored arms, forcing the ML agent to pick those 
  arms to reduce their UCB radius.

Assuming our rewards are in the range $[0,1]$, the $\text{UCB}(a)$ is:

$$\text{UCB}(a) = \bar{\mu}(a) + \sqrt{\frac{2\ln(T)}{N}}$$

where again $N$ is the number of times that arm $a$ is pulled, and $T$ is set to the number of arms pulled so far by the agent regardless which.

```python
class UCB1(MAB): # it extends class MAB to implement UCB

    def __init__(self, beta=1.0):
        '''Constructor.'''
        super().__init__()
        self.beta = beta
        self.overall_total_count = 0
        self.ucb = 0

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
```

## Outcomes<a name=outcomes></a>

The following shows some statistics of the learning. As can be seen, the average rewards for all arms are quite similar. Based on the environment, we know that some ads have low theoretical click rate. But since we use the UCB as the reward, they are given the benefit of doubt with a higher UCB radius, and hence their reward is artificially improved. The UCB radius is shown under `UCB raduis` in the animation.

```console
Testing UCB MAB

 Ad      Average  UCB   Ad shown
type      reward radius to users
--------------------------------
> toys      0.55  0.31  [==] 96
> cars      0.61  0.18  [=========] 365
> sports    0.61  0.12 *[=======================] 941
> holidays  0.61  0.17  [==========] 419
> foods     0.60  0.24  [====] 174

Click rate =  0.35
Regret = 114.35

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 2000
Number of clicks = 709
Click rate = 35.45%
Theoretical best click rate = 40.00%
```

Unlike the simple MAB where the learning can be highly influenced by the short-term bias in the environment, UCB can self-correct this bias by offseting the effect using UCB radius. Thus in the simple MAB, the exploration rate must not be too low to avoid being influenced by the short-term bias in the environment, UCB can be operated with a low exploration rate. When all arms are sufficiently explored and all UCB radii are equally low, the best arm will be revealed.

## What's Next?<a name=next></a>

In the previous chapter, we introduce MAB and demonstrated its operation using a primitive MAB. This chapter discusses the classical UCB which aims to avoid missing potential good arms due to short-term bias in the environment. 

Imagine if the ML agent can estimate the distribution of each arm rather than just an upper bound, it will have much more information to make better decision. Thompson Sampling technique provides a means to estimate the distribution of an arm by continuingly shaping the estimated distribution using observed rewards. How? This will be our topic in the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts).
