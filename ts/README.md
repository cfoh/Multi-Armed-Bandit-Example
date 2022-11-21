# Thompson Sampling Technique

<table>
<tr><td colspan="2"><b>
Chapter 4: Thompson Sampling Technique
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
  <img src="https://user-images.githubusercontent.com/51439829/200311722-19f95e88-9583-44b4-aa54-fb9d72c69e1f.gif" width="400">
</td>
<tr><td colspan="2">

In the demo, the ML agent attempts to estimate the distribution of each arm based on observed rewards. During exploitation, a random sample is drawn from each distribution and the agent exploits the arm that produced the best sample among all. The `Last drawn` column shows the value sampled from each arm.
<br>
Press `[F5]` to restart the demo.
</td>
<tr><td colspan="2">
<b>Contents</b><br>
<ul>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example">Chapter 1: Simple Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb">Chapter 2: Upper Confidence Bound (UCB) Algorithm</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax">Chapter 3: Boltzmann Exploration (Softmax)</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts">Chapter 4: Thompson Sampling Technique</a>&nbsp;<img src="https://img.shields.io/badge/This-Chapter-blue"></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab">Chapter 5: Contextual Multi Armed Bandit</a></li>
<li><a href="https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2">Chapter 6: Contextual Multi Armed Bandit (more)</a></li>
</ul>
</td></tr>
</table>

## Introduction<a name=intro></a>

To understand Thompson Sampling, we need to describe a few concepts:
- A prior distribution $P(\theta)$ is our intuition of the parameter $\theta$ 
  describing the outcome distribution without inspecting any outcome.
- A posterior distribution $P(\theta|y)$ is an update of the prior after
  inspecting some outcome $y$.
- The likelihood function $P(y|\theta)$ describes the likelihood of some outcome
  $y$ given the parameter $\theta$.

The above forms the relationship (i.e. Bayes' Theorem): 

$$P(\theta|y) = \frac{P(y|\theta) P(\theta)}{P(y)}$$ 

which is $P(\theta|y) \propto P(y|\theta) P(\theta)$ since $P(y)$ is just some constant. Our goal is to obtain the posterior by updating the prior using observing outcomes. If the posterior gives the same distribution as the prior, then we can use the posterior as the next prior to repeat the update process over and over again. Hence we need the posterior to be in the same probability distribution family as the prior.

Back to our digital advertising example, since the outcome is either a click or not, the reward follows Bernoulli distribution, which describes the likelihood of an outcome. With the likelihood of Bernoulli, it is shown that its conjugate prior is a Beta distribution (see Example Section of [this document](https://en.wikipedia.org/wiki/Conjugate_prior) for the proof). In other words, if we pick $\mbox{Beta}(\alpha,\beta)$ to be our initial reward distribution estimation, by updating with an observe Bernoulli outcome $s$, we can show that the posterior is $\mbox{Beta}(\alpha+s,\beta+(1-s))$ which is another Beta distribution, where $\alpha$ and $\beta$ are two hyperparameters of Beta distribution. The above provides us a facility to repeatedly update and refine the Beta distribution based on the outcome.

Lastly, we need to put the established posterior distribution to good use. We do this by sampling posterior distribution. Precisely, the ML agent draws a sample from the posterior distribution of each arm, and then it exploits the arm that produces the highest value of sample among all.

## Implementation<a name=codes></a>

The implementation of Thompson Sampling technique for Bernoulli reward is quite straightforward. We first construct Beta(1,1) as our initial prior. Beta(1,1) is also a uniform distribution which makes sense, as we want to begin without any bias on the reward outcome. 

As described in the earlier section, the posterior is another Beta distribution with some update on the hyperparameters $\alpha$ and $\beta$. When an arm is pulled and the corresponding reward, $s$, is observed, we can update the prior to produce the posterior distribution by the following. It will be then used as the next prior for further update:

$$\text{Beta}(\alpha,\beta) \leftarrow \text{Beta}(\alpha+s,\beta+(1-s)).$$

The following is the implementation:

```python
class TS:

    def __init__(self):
        '''Constructor.'''
        self.total_count = {}
        self.alpha = {}
        self.beta = {}
        self.last_drawn = {}

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
```

## Outcomes<a name=outcomes></a>

We show the result of a Thompson Sampling run in the following. The column under `Last drawn` records the last sample drawn from the estimated reward distribution. In this run, the click rate is quite close to the optimal one.

```console
Testing Multi-armed Bandit with Thompson Sampling technique

 Ad      Average  Last   Ad shown
type      reward  drawn  to users
-------------------------------
&gt; toys      0.16  0.23   [==] 91
&gt; cars      0.35  0.36   [=======] 293
&gt; sports    0.42  0.39  *[=================================] 1351
&gt; holidays  0.37  0.35   [====] 178
&gt; foods     0.28  0.18   [==] 87

Click rate =  0.39
Regret = 78.55

Strategy: Epsilon Greedy, epsilon = 0.15
Number of users = 2000
Number of clicks = 772
Click rate = 38.60%
Theoretical best click rate = 40.00%
```

We can also visualize the evolution of the estimated reward distribution. The following animation illustrates how the estimated reward distribution is refined over the course of learning.

<img src="https://user-images.githubusercontent.com/51439829/200311875-ffd3d50f-b36c-441c-9281-d8408a7ea5df.gif" width="400"/>

## What's Next?<a name=next></a>

In this chapter, we demonstrated using Thompson Sampling technique to improve decision making in MAB. We focused on our example of digital advertizing where the outcome follows a Bernoulli ditribution, and we used Beta distribution as our prior with $\alpha$ and $\beta$ being the hyperparameters to update.

Your problem may be different where the outcome may follow a different distribution. Then you need to find an appripriate prior, identify the hyperparameters and the method to update the prior. You may check this [wikipedia article](https://en.wikipedia.org/wiki/Conjugate_prior) which provides a table of conjugate distributions showing that for a particular likelihood function, which prior should be used, what are the involved hyperparameters, and how to update the prior to produce the posterior distribution.

There are many other MAB variants, each with its own pros and cons. Some commonly discussed techniques are UCB2, UCB1-Tuned, Contextual Bandits and LinUCB, etc. [Next](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab), we shall introduce the concept of `contexts` and explain how this can further improve the learning.
