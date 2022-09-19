# Contextual Multi-Armed Bandit Example

## Introduction

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

We first need to define the behaviour of users. This is the 
environment logic defining how a user responds to an offered 
advertisement. The logic behind it is not known to the ML agent,
and it is the task of the ML agent to learn and establish the
relationship. The ML agent knows the user profile, and its job
is to discover which advertisement is most attactive to the
user.

The behaviour of users are described in the following table. 
It shows the likelihood of each age clicking different types
of advertisements.
```
+-------------------+--------------------------------------+
|                   |              Age group               |
| Ad Type           |  <25    26-35   36-45   46-55  >55   |
+-------------------+--------------------------------------+
| Toys & Games      |  80%     15%      5%     5%     5%   |
| Cars              |   5%     50%     15%    10%     5%   |
| Sports            |   5%     10%     40%    25%    10%   |
| Holiday Packages  |   5%     20%     20%    50%    20%   |
| Foods & Health    |   5%      5%     20%    10%    60%   |
+-------------------+--------------------------------------+
```

With Contextual MAB, our setup is:
- the `context` is the age group
- the `arms` are advertisement type to offer
- the `reward` is 100 if click is registered, otherwise 0

## Outcomes

We measure the effectiveness of our strategy using `click rate`. It is the percentage that a user will click and explore the offered advertisement. By setting epsilon to 0.15, we achieve around 48% of click rate. With the user behaviour, the theoretical best click rate we can achieve is 56%. As you can see, we are not far from the theoretical best.

```
Epsilon = 0.15
Number of users = 10000
Number of clicks = 4803
Click rate = 48.03%
Theoretical best click rate = 56.00%

Age group   Ad to offer    Expected CMAB reward
===============================================
    0         toys              79.41
    1         cars              50.77
    2         sports            39.00
    3         holidays          49.32
    4         foods             59.99
```

The theoretical result is calculated based on the assumption that we know the user behaviour matrix. As a result, knowing the user age group, we will always offer the most attractive advertisement to the user. For example, for Age Group 0 (or <25), we'll always show advertisement related to "Toys" as we know that 80% of the users will click and explore the advertisement, and thus our success rate is 80% too. So, the overall successful click rate is the average over all user age groups, and this gives

<p align="center">
$\frac{80+50+40+50+60}{5} = 56$.
</p>

## Plots

We can also see how click rate evolves over the time for each user age group. 

<img src="https://user-images.githubusercontent.com/51439829/191068018-893da713-03cb-49e1-94c2-b0f47f8adcc2.png" height="300">

See below, if we set `epsilon = 1.0`, we essentially force the ML agent to operate in 100% exploration. During exploration, the ML agent simply randomly picks a type of advertisements. The click rate is thus 20% (why?).

<img src="https://user-images.githubusercontent.com/51439829/191068591-0055e7ab-a9db-4465-a207-ffffe189db3e.png" height="300">

