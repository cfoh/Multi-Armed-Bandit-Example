# Chapter 3: Boltzmann Exploration (Softmax)

<table>
<tr><td colspan="2"><b>
Chapter 3: Boltzmann Exploration (Softmax)
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
The above demo shows the empirical average and the corresponding probability for each arm. Because the arm selection is based on the corresponding probability, the agent will not always choose the arm with the maximum probability. As can be seen, the agent occasionally tries other arms that don't produce the highest probability.
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
</ul>
</td></tr>
</table>

## Introduction<a name=intro></a>

In the previous chapters, we see the ML agent greedily chooses the arm with the highest average reward. While choosing the highest average reward seems to be the best option, it may miss other options that are just below the best, and one of these options may actually be the best but just suffers from short-term bias.

Rather than focusing on the best, Boltzmann Exploration first establishes a weight for each arm based on its empirical average reward compared to all others, then pick an arm based on the weights. viewuses a Pick an arm with a probability proportional to is average reward. The following is how it establishes the weight (or probability). Let there be $N$ arms and $\mu_n$ be the empirical mean reward of $n$-th arm. The probability for the agent to pick arm $a$ follows the following softmax function where $\tau$ is a hyperparameter scaling mean rewards:

$$P(a) = \frac{\exp(\frac{\mu_a}{\tau})}{\sum_{n=1}^{N}\exp(\frac{\mu_n}{\tau})}$$

## Implementation<a name=codes></a>

The implementation of Boltzmann Exploration is fairly easy. Python already has a random choice function based on input weights. We only need to produce the weight for each arm. Since the input weights are not a probability distribution function, they do not need to sum to 1, we can simply provide the nominator as the weight.

```python
class SoftMax(MAB):
    '''
    Boltzmann Exploration (Softmax).
    '''

    def __init__(self, tau=1.0):
        '''Constructor.'''
        super().__init__()
        self.tau = tau

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
```

## Outcomes<a name=outcomes></a>

For softmax exploration, we're also interested in the weight for each arm. Thus, in the demo, we also show the weight which is also the probability that an arm will be picked. In the simulation, we set $\tau=0.05$. The value is set arbitrarily. Setting it too large will flatten the distribution curve while setting it too small will sharpen the distribtion curve and create large contrast among the probabilities. The large contrast will cause the agent to focus on the highest mean reward, making the decision hard again.

```console
Testing Boltzmann Exploration (Softmax)

Ad_type   Reward  Weight Ad_shown_to_users
------------------------------------------
> toys      0.01  0.00   [=] 73
> cars      0.26  0.05   [===] 128
> sports    0.40  0.77  *[==============================] 1211
> holidays  0.32  0.17   [===========] 475
> foods     0.19  0.01   [==] 108

Click rate =  0.34
Regret = 76.65

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 2000
Number of clicks = 689
Click rate = 34.45%
Theoretical best click rate = 40.00%
```

## What's Next?<a name=next></a>

Boltzmann Exploration is not the only technique avoiding making a hard decision. Thompson Sampling too chooses the option probabilistically to exploit. Instead of deriving a probability for each arm. Thompson Sampling attempts to derive a distribution for each arm, and then use a sample from the distribution to choose the best option. How? This will be our topic in the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts).
