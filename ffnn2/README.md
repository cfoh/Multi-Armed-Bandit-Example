# Chapter 9: Deep Learning Based Bandit for C-MAB

### Contents

- [Chapter 1: Simple Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example)
- [Chapter 2: Upper Confidence Bound (UCB) Algorithm](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb)
- [Chapter 3: Boltzmann Exploration (Softmax)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax)
- [Chapter 4: Thompson Sampling Technique](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts)
- [Chapter 5: Contextual Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab)
- [Chapter 6: Contextual Multi Armed Bandit (more)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2)
- [Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn)
- [Chapter 8: Linear Bandit and LinUCB](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/linucb)
- Chapter 9: Deep Learning Based Bandit for C-MAB ![This-Chapter](https://img.shields.io/badge/This-Chapter-blue)
  - [Introduction](#intro)
  - [Implementation](#codes)
  - [The Outcomes](#outcomes)

## Introduction<a name=intro></a>

In Chapter 7, we first introduced using Feed Forward Neural Network (FFNN) in Contextual Multi-Armed Bandit (C-MAB). In the chapter, we show that how FFNN can be used to replace the action-value table in MAB. Similar to a table-driven MAB, we use discretized contexts where we *somehow* know that a certain user group has similar behaviour so that they can be grouped together to form a unique context.

However, such a grouping is often not immediately available in the real-world. Clustering users into similar behaviour requires further analysis such as unsupervised learning techniques, and this is out of the scope of our study here.

In the last chapter, we show that if the context and its reward forms a linear relationship, then we can apply LinUCB. As LinUCB uses linear regression, the contexts can be a continuous value. However, the assumption that the context and its reward forming a linear relationship is sometimes considered impractical. As we have shown in the example, because the ad `Sports` exhibits a bell-shape reward, LinUCB struggled to predict its reward at the peak of the reward curve and failed to make accurate recommendation.

To overcome the limitation, we can use neural network for the regression task. The challenge of using neural network is the slower operation and the need for huge amount of data for training. Nevertheless, we shall show how we can use neural network for the regression talk in C-MAB.

## Implementation<a name=codes></a>

We shall reuse the example given in LinUCB as follows. The table defines environment or the **true probability** that a user with a given feature will pick a particular ad:

```
  The Environment
+-------------------+--------------------------------------+
|                   |              Age group               |
| Ad Type           |  <25    26-35   36-45   46-55  >=55  |
+-------------------+--------------------------------------+
| Toys & Games      |  76%     64%     52%    40%    28%   |
| Cars              |  10%     25%     40%    55%    70%   |
| Sports            |  15%     30%     70%    30%    15%   |
| Foods & Health    |   5%     25%     25%    40%    80%   |
+-------------------+--------------------------------------+
```

We build a neural network with one input since the context is simple the age of users. The neural network has 4 outputs to predict the reward of each ad `Toys`, `Cars`, `Sports` and `Foods`. The following is the model implementation:

```python
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=num_features))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_rewards, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
```

As usual, the training and model update is done at the end of each episode when sufficient data are collected. We set the length of our episode to be 1000 data collection. During the episode, we store the collected data in a memory buffer. The format of each data point is `(feature, arm, reward)` or in our case, it is `(user_age, ad_offered, reward_observed)`. At the end of each episode, the memory buffer contains 1000 data points.

The data in the memory is not yet suitable for training. For training, we need input-output relationship, i.e. feature as the input, and rewards for all ads as the outputs. As can be seen, each data point in our memory only contains the reward of the ad offered. We do not know what will be the reward if the system offered the other ads as the system can only offer one ad to the user. Fortunately, we can use the current model to make prediction. As a result, we can transform the collected data in the memory buffer into training data suitable for training:
```
   memory buffer                    train data
(feature,arm,reward)         (feature,list_of_all_rewards)
--------------------         -----------------------------
[((age1,),'toys',toy1),      [((age1,), [toy1, car_pred, sport_pred, food_pred]),
 ((age2,),'foods',food2),     ((age2,), [toy_pred, car_pred, sport_pred, food2]),
    ...                       ...
 ((age_k,),'cars',car_k)]     ((age_k,), [toy_pred, car_k, sport_pred, food_pred])]
```
where `toy_pred`, `car_pred`, `sport_pred` and `food_pred` are the missing rewards filled by model prediction.

With `train_data` prepared, we can proceed with the model training by:
```python
X = np.array([list(d[0]) for d in train_data], dtype=np.float32)
y = np.array([d[1] for d in train_data], dtype=np.float32)
model.fit(X, y, batch_size=30, epochs=5, verbose=0)
```

Note that the feature is a tuple. In our example, it is a single element tuple containing the age of the user.

## The Outcomes<a name=outcomes></a>

We setup a scenario to recommend ads to 50000 users. For each user, we pick a type of ad and recommend 20 ads of that type. The user will have an opportunity to click any, and we measure how many ads does the user click or like.

Our reward for each user is the click through rate (or simply click rate) which is measured by the percentage that the user has clicked out of the 20 offered ads. If the user clicked all, then the click rate is 1.0. If the user clicked into 5 of the ads to explore, the click rate is 5/20 = 0.25.

In the following plot, we compare the ground truth (the solid lines) and the predicted rewards (the dotted lines). Unlike LinUCB, FFNN is flexible to produce a shape of curve that best represents the seen data. The data are used for training, the more accurate the predict becomes, provided that the FFNN model is sufficiently large to produce the complex shape.

![ffnn2-ground-truth](https://github.com/cfoh/Multi-Armed-Bandit-Example/blob/main/ffnn2/fig-ffnn2-ground-truth-vs-prediction.png)

The following table lists the outcome for all features or ages. Recall that LinUCB failed to recommend `Sports` as its reward curve is not linear. With FFNN, the issue is overcome. As can be seen, it can now accurately recommend `Sports` most of the time, although it missed some other ads. Its performance can be further improved by undergoing more training.

```
+----------------------------------------------------------------------+
| age |   best  |  expected  | suggested |    predicted click rate     |
+     |    ad   | click rate |     ad    | [toys, cars, sports, foods] |
+-----+---------+------------------------------------------------------+
|  5  |  toys   |    0.76    |   toys    |  [0.74, 0.10, 0.09, 0.01]   |
|  6  |  toys   |    0.76    |   toys    |  [0.74, 0.10, 0.10, 0.01]   |
|  7  |  toys   |    0.76    |   toys    |  [0.74, 0.10, 0.10, 0.02]   |
|  8  |  toys   |    0.76    |   toys    |  [0.74, 0.10, 0.11, 0.02]   |
|  9  |  toys   |    0.76    |   toys    |  [0.73, 0.09, 0.12, 0.02]   |
| 10  |  toys   |    0.76    |   toys    |  [0.73, 0.09, 0.13, 0.02]   |
| 11  |  toys   |    0.76    |   toys    |  [0.73, 0.09, 0.14, 0.02]   |
| 12  |  toys   |    0.76    |   toys    |  [0.72, 0.09, 0.15, 0.02]   |
| 13  |  toys   |    0.76    |   toys    |  [0.72, 0.10, 0.16, 0.03]   |
| 14  |  toys   |    0.76    |   toys    |  [0.71, 0.10, 0.16, 0.03]   |
| 15  |  toys   |    0.76    |   toys    |  [0.71, 0.10, 0.17, 0.03]   |
| 16  |  toys   |    0.76    |   toys    |  [0.72, 0.10, 0.18, 0.03]   |
| 17  |  toys   |    0.76    |   toys    |  [0.72, 0.10, 0.18, 0.04]   |
| 18  |  toys   |    0.76    |   toys    |  [0.72, 0.10, 0.19, 0.04]   |
| 19  |  toys   |    0.76    |   toys    |  [0.72, 0.11, 0.19, 0.05]   |
| 20  |  toys   |    0.76    |   toys    |  [0.71, 0.11, 0.18, 0.06]   |
| 21  |  toys   |    0.76    |   toys    |  [0.71, 0.12, 0.18, 0.07]   |
| 22  |  toys   |    0.76    |   toys    |  [0.70, 0.13, 0.18, 0.09]   |
| 23  |  toys   |    0.76    |   toys    |  [0.69, 0.14, 0.18, 0.11]   |
| 24  |  toys   |    0.76    |   toys    |  [0.68, 0.15, 0.19, 0.13]   |
| 25  |  toys   |    0.64    |   toys    |  [0.67, 0.17, 0.19, 0.16]   |
| 26  |  toys   |    0.64    |   toys    |  [0.66, 0.18, 0.19, 0.19]   |
| 27  |  toys   |    0.64    |   toys    |  [0.65, 0.20, 0.19, 0.22]   |
| 28  |  toys   |    0.64    |   toys    |  [0.64, 0.22, 0.22, 0.22]   |
| 29  |  toys   |    0.64    |   toys    |  [0.63, 0.23, 0.25, 0.22]   |
| 30  |  toys   |    0.64    |   toys    |  [0.62, 0.25, 0.29, 0.21]   |
| 31  |  toys   |    0.64    |   toys    |  [0.61, 0.27, 0.33, 0.21]   |
| 32  |  toys   |    0.64    |   toys    |  [0.60, 0.29, 0.38, 0.20]   |
| 33  |  toys   |    0.64    |   toys    |  [0.58, 0.31, 0.43, 0.20]   |
| 34  |  toys   |    0.64    |   toys    |  [0.57, 0.33, 0.47, 0.20]   |
| 35  | sports  |    0.70    |   toys    |  [0.56, 0.36, 0.52, 0.19]   |  missed
| 36  | sports  |    0.70    |  sports   |  [0.55, 0.38, 0.57, 0.19]   |
| 37  | sports  |    0.70    |  sports   |  [0.54, 0.40, 0.62, 0.19]   |
| 38  | sports  |    0.70    |  sports   |  [0.53, 0.43, 0.66, 0.18]   |
| 39  | sports  |    0.70    |  sports   |  [0.52, 0.45, 0.70, 0.18]   |
| 40  | sports  |    0.70    |  sports   |  [0.51, 0.46, 0.67, 0.20]   |
| 41  | sports  |    0.70    |  sports   |  [0.50, 0.47, 0.64, 0.22]   |
| 42  | sports  |    0.70    |  sports   |  [0.49, 0.48, 0.61, 0.23]   |
| 43  | sports  |    0.70    |  sports   |  [0.48, 0.49, 0.58, 0.25]   |
| 44  | sports  |    0.70    |  sports   |  [0.46, 0.51, 0.55, 0.27]   |
| 45  |  cars   |    0.55    |  sports   |  [0.45, 0.52, 0.52, 0.29]   |  missed
| 46  |  cars   |    0.55    |   cars    |  [0.44, 0.53, 0.49, 0.32]   |
| 47  |  cars   |    0.55    |   cars    |  [0.43, 0.54, 0.46, 0.34]   |
| 48  |  cars   |    0.55    |   cars    |  [0.42, 0.55, 0.43, 0.36]   |
| 49  |  cars   |    0.55    |   cars    |  [0.41, 0.56, 0.40, 0.39]   |
| 50  |  cars   |    0.55    |   cars    |  [0.40, 0.57, 0.37, 0.41]   |
| 51  |  cars   |    0.55    |   cars    |  [0.39, 0.58, 0.34, 0.44]   |
| 52  |  cars   |    0.55    |   cars    |  [0.38, 0.59, 0.31, 0.46]   |
| 53  |  cars   |    0.55    |   cars    |  [0.37, 0.60, 0.29, 0.49]   |
| 54  |  cars   |    0.55    |   cars    |  [0.36, 0.61, 0.26, 0.51]   |
| 55  |  foods  |    0.80    |   cars    |  [0.35, 0.62, 0.24, 0.54]   |  missed
| 56  |  foods  |    0.80    |   cars    |  [0.34, 0.64, 0.22, 0.56]   |  missed
| 57  |  foods  |    0.80    |   cars    |  [0.33, 0.65, 0.20, 0.59]   |  missed
| 58  |  foods  |    0.80    |   cars    |  [0.32, 0.66, 0.18, 0.61]   |  missed
| 59  |  foods  |    0.80    |   cars    |  [0.31, 0.67, 0.16, 0.64]   |  missed
| 60  |  foods  |    0.80    |   cars    |  [0.30, 0.68, 0.15, 0.66]   |  missed
| 61  |  foods  |    0.80    |   cars    |  [0.29, 0.68, 0.13, 0.68]   |  missed
| 62  |  foods  |    0.80    |   foods   |  [0.28, 0.69, 0.12, 0.71]   |
| 63  |  foods  |    0.80    |   foods   |  [0.27, 0.70, 0.11, 0.73]   |
| 64  |  foods  |    0.80    |   foods   |  [0.26, 0.71, 0.09, 0.75]   |
| 65  |  foods  |    0.80    |   foods   |  [0.25, 0.72, 0.08, 0.77]   |
| 66  |  foods  |    0.80    |   foods   |  [0.24, 0.73, 0.08, 0.78]   |
| 67  |  foods  |    0.80    |   foods   |  [0.24, 0.74, 0.07, 0.80]   |
| 68  |  foods  |    0.80    |   foods   |  [0.23, 0.75, 0.06, 0.82]   |
| 69  |  foods  |    0.80    |   foods   |  [0.22, 0.76, 0.05, 0.83]   |
| 70  |  foods  |    0.80    |   foods   |  [0.21, 0.76, 0.05, 0.85]   |
+-----+---------+------------------------------------------------------+
```
