# Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)

### Contents

- [Chapter 1: Simple Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example)
- [Chapter 2: Upper Confidence Bound (UCB) Algorithm](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb)
- [Chapter 3: Boltzmann Exploration (Softmax)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax)
- [Chapter 4: Thompson Sampling Technique](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts)
- [Chapter 5: Contextual Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab)
- [Chapter 6: Contextual Multi Armed Bandit (more)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2)
- Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN) ![This-Chapter](https://img.shields.io/badge/This-Chapter-blue)
  - [Introduction](#intro)
  - [Implementation](#code)
  - [The Outcomes](#outcomes)
  - [What's Next?](#next)
- [Chapter 8: Linear Bandit and LinUCB](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/linucb)
- [Chapter 9: Deep Learning Based Bandit for C-MAB](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn2)


## Introduction<a name=intro></a>

In Contextual Multi-Armed Bandit (C-MAB), the machine learning (ML) agent uses a table to store values of actions for different contexts. This context-action table can be replaced by a feed forward neural network (FFNN). There are several advantages to use a FFNN instead of a table. FFNN can better handle context with a huge space, and can deal with context containing continuous values. Besides, FFNN can natively predict action values for unseen contexts whereas table-driven C-MAB cannot and it must use a separate technique such as KNN to infer the action values. However, FFNN requires more data to get trained.

## Implementation<a name=codes></a>

Here we use tensorflow library to create a feed forward neural network (FFNN). Our FFNN takes context as the input and produces values of actions. 
- Input (context): Since we have 5 user age groups, we use 5 binary inputs each represents 
  a user age group.
- Output (action value): We have 5 ad types, hence we use 5 outputs each represents a
  specific ad type, and the output value is the predicted reward for the corresponding ad type.

To train the FFNN, the outcomes of each episode which contains a number of user services are first collected in a memory. At the end of the episode, the memory is used to train the FFNN. However, the raw data in the memory is not suitable for training, as the reward for each action is given by either 0 or 1 rather than a probability. FFNN is unable to measure the probability directly from the memory. For example, for the following data in the memory
```
context=[1,0,0,0,0], action=2, reward=0
context=[1,0,0,0,0], action=2, reward=1
context=[1,0,0,0,0], action=2, reward=1
context=[1,0,0,0,0], action=2, reward=0
context=[1,0,0,0,0], action=2, reward=1
...
```
and we need to summarize them into a single data record for training as follows
```
context=[1,0,0,0,0], action=2, reward=0.6
```

We use the following FFNN to train the data:
```python
model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=num_contexts))
model.add(Dense(units=50, activation='tanh'))
model.add(Dense(units=num_actions))
model.compile(loss='mean_squared_error', optimizer='adam')
```

There are many ways we can incorporate the data from the memory into FFNN, here we use exponential smoothing:
```python
y_value[action] = (1-alpha)*y_value[action] + alpha*reward[context][actioin]
```
where `reward[context][actioin]` is the summarized probability for the given `context` & `action`, and `alpha` is the learning rate.

## Outcomes<a name=outcomes></a>

The program shows the following tables for each episode:
- Fact: this is the fact which is what the environment is used to produce reward
- This episode: this is the data collected for this episode, it contains the summarized
  rewards for each pair of context and action
- C-MAB table: this is the context-action table recorded by C-MAB algorithm, it is shown here
  for comparison
- FFNN prediction: this is the context-action values predicted by FFNN after the training

The following shows an example of the outcome:
```
              Fact                  This episode              C-MAB table            FFNN prediction
c=0: [0.80,0.05,0.15,0.05,0.05][0.81,0.00,0.15,0.00,0.06][0.83,0.06,0.15,0.03,0.05][0.78,0.07,0.12,0.03,0.05]
c=1: [0.15,0.50,0.30,0.20,0.25][0.18,0.30,0.31,0.28,0.43][0.14,0.44,0.31,0.21,0.23][0.09,0.49,0.28,0.20,0.23]
c=2: [0.10,0.30,0.40,0.35,0.25][0.07,0.32,0.24,0.40,0.32][0.07,0.27,0.41,0.35,0.24][0.06,0.30,0.47,0.36,0.22]
c=3: [0.05,0.15,0.30,0.50,0.40][0.00,0.12,0.28,0.44,0.31][0.04,0.20,0.31,0.54,0.41][0.03,0.15,0.30,0.55,0.39]
c=4: [0.05,0.10,0.30,0.50,0.60][0.00,0.19,0.41,0.56,0.53][0.06,0.11,0.34,0.48,0.53][0.05,0.12,0.34,0.43,0.51]
```

As can be seen, C-MAB table tracks closely to the Fact while FFNN gives good but not excellent prediction. Since the decision is based on selecting the action with the highest action value among all for each context, accuracy of action values are not critical as long as the highest action value is the best action based on the Fact. We see that both C-MAB & FFNN are able to accurately pick the best action for each context.

We conclude that since this is a very simple problem, use of FFNN is unnecessary. Table driven C-MAB is more than adequate to handle this problem. The purpose of this example is to illustrate how FFNN can be used to replace the context-action table in C-MAB.

## What's Next?<a name=next></a>

In this example, the contexts are clearly partitioned which make the establishment of context-action table relative straightforward. As a result, replacing it with FFNN is not a clever move indeed, as FFNN requires more data for training to achieve accurate prediction. In the real-world applications, how contexts are partitioned is often unknown. In this example, we may only know that the behaviour of a user depends on his/her age, but we do not know how to partition them into groups to describe the contexts. The clustering may require additional analysis using techniques such as unsupervised learning.

Fortunately, if we know that the context-reward has a linear relationship, we can use linear regression to make prediction of the rewards. In the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/linucb/README.md), we shall explore LinUCB which is an effective technique when context-reward forms a linear relationship.
