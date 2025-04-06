# Chapter 8: Linear Bandit and LinUCB

### Contents

- [Chapter 1: Simple Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example)
- [Chapter 2: Upper Confidence Bound (UCB) Algorithm](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ucb)
- [Chapter 3: Boltzmann Exploration (Softmax)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/smax)
- [Chapter 4: Thompson Sampling Technique](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ts)
- [Chapter 5: Contextual Multi Armed Bandit](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab)
- [Chapter 6: Contextual Multi Armed Bandit (more)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/cmab2)
- [Chapter 7: Implementing C-MAB using Feed Forward Neural Network (FFNN)](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn)
- Chapter 8: Linear Bandit and LinUCB ![This-Chapter](https://img.shields.io/badge/This-Chapter-blue)
  - [Introduction](#intro)
  - [LinUCB](#linucb)
  - [A toy example](#example)
  - [Implementation](#codes)
  - [The Outcomes](#outcomes)
  - [What's Next?](#next)
- [Chapter 9: Deep Learning Based Bandit for C-MAB](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn2)


## Introduction<a name=intro></a>

Linear Upper Confidence Bound (LinUCB) is a **contextual bandit algorithm** first introduced for **personalized recommendations** by Li *et al.* in their work [here](https://arxiv.org/pdf/1003.0146.pdf). The main assuming is that at any round $t$, the expected reward of an arm $a$ is linear in its feature $x_{t,a}$ with some unknown coefficient vector $\theta_a$. The authors called this model *disjoint* since the arms are not shared among different arms. In other words, each arm has its own independent unknown coefficient vector.

### Linear Regression

Consider that we want to predict the house price in an area. It is found that the house price is linear to the number of bedrooms. Moreover, it is also found that the house price is linear to the size of the garden. Given these two features (number of rooms and size of the garden), we want to establish a relationship between the house prices given the two features.

Given the linear relationship, we can use the following formula:

$$y = β_0 + β_1 x_1 + β_2 x_2 + \epsilon$$

where $x_1$ and $x_2$ are the number of rooms and the size of the garden, respectively, and $\epsilon$ is the error term following a certain unknown distribution. If we collect many records of $(x_1, x_2, y)$, we can apply linear regression to estimate the coefficients $β_0, β_1$ and $β_2$.

We often use matrix notation to describe the linear regression problem as:

$$y = X β + \epsilon$$

where $X$ is the input matrix (a.k.a design matrix), $y$ is the target vector, and $\epsilon$ is the error value vector. Consider that we have collected 100 samples, then $X$ is a 100-by-2 matrix, and $y$ and $\epsilon$ are a column vector with 100 elements.

### Least-Squares Estimation

With some collected data, we can apply least-squares estimation to estimate the coefficients $β$. The idea is to search for the coefficients that produces a linear function best representing the given input data. Essentially, we measure how close the function predicts the data, that is, the distance between the actual value $y_i$ and the estimated value $x_i^\top \hat{β}$ for the $i$-th sample given estimated coefficients $\hat{β}$. The objective of least-squares estimation is find $β^*$ such that the sum of the distances is minimized:

$$β^* = \underset{\hat{β}}{\arg\min} \sum_{i=1}^{n} \left( x_i^\top \hat{β} - y_i \right)^2$$

It turns out that the coefficient vector can be directly solved by ([see here for the derivation](https://en.wikipedia.org/wiki/Linear_regression#Estimation_methods)):

$$β = (X^\top X)^{-1} X^\top y$$

where $X^\top$ is the transpose of $X$.

### Ridge Regression

Ridge Regression is design for a system where the independent variables are highly correlated. It introduces a method of regularization controlled by a parameter $\lambda$:

$$\hat{β} = \underset{β}{\arg\min} \sum_{i=1}^{n} \left( x_i^\top β - y_i \right)^2 + \lambda \left\| β \right\|^2$$

With Ridge Regression, we see that the solution favors smaller coefficient values. Similar to least-squares estimation, Ridge Regression also possesses a closed-form solution:

$$β = (X^\top X + \lambda I)^{-1} X^\top y$$

where $I$ is an identity matrix. The following is the implementation of Ridge Regression (see also the full source code [here](https://)):

```python
#########################################################
## Closed-form Ridge regression solver
##
## The coefficient vector β = (X^T X + λI)^-1 X^T y
## where:
## - X is the input matrix with shape (n_samples, n_features)
## - y is the target vector with shape (n_samples)
## - λ is the regularization parameter
## - I is the identity matrix of shape (n_features, n_features)
## - X^T is the transpose of X
## Note:
## - if the intercept β0 is not zero, the input X should
##   be augmented with a column of ones, 
##   i.e. X = [1, x1, x2, ...]
#########################################################

def ridge_regression(X, y, lambda_ridge):
    _, dim = X.shape
    coeffs = np.linalg.inv(X.T @ X + lambda_ridge * np.eye(dim)) @ X.T @ y
    return coeffs
```

### Online Solver

The closed-form solution allows direct and efficient estimation of the coefficients. In this setup, all the collected data must be available to be used for solving. If we only have partial data, we can perform online estimation where we estimate and update the coefficients progressively as data arrive.

The online solver is similar to the closed-form formulation. We first initialize the following:

$$A = X^\top X = 0$$
$$b = X^\top y = 0$$

For each input $(\vec{x}_i,y_i)$, we perform the following update:

$$A \leftarrow A + \vec{x}_i \vec{x}_i^\top$$
$$b \leftarrow b + y_i \vec{x}_i$$

We then compute the coefficient matrix and estimate the $y$ value by:

$$\hat{β} = (A + λI)^{-1} b$$
$$\hat{y} = \hat{β}^\top \vec{x}_i$$

The following is the implementation of online Ridge Regression (see also the full source code [here](https://))::

```python
#########################################################
## Online Ridge regression solver
## 
## Initialize X^T X = 0 and X^T y = 0
## for each sample xi, yi:
## - update X^T X += xi @ xi.T
## - update X^T y += yi * xi
## - compute coefficients β = (X^T X + λI)^-1 X^T y
## where:
## - xi is the i-th sample (column vector)
## - yi is the corresponding target value
## - λ is the regularization parameter
## - I is the identity matrix of shape (dim, dim)
## - β is the vector of coefficients (intercept and slopes)
## - X^T is the transpose of X
## Note: 
## - the intercept term is included in the system, i.e. if there are 2 features,
##   the dimension of the problem is 3 (2 features + intercept), and the coefficients
##   vector contains β0, β1 and β2 for the intercept, feature 1 and 2 respectively.
#########################################################

class OnlineRidgeRegression:

    def __init__(self, num_features, lambda_ridge):
        self.lambda_ridge = lambda_ridge
        self.dim = num_features+1                  # dimension of the problem
        self.XTX = np.zeros((self.dim, self.dim))  # initialize X^T X matrix
        self.XTy = np.zeros(self.dim)              # initialize X^T y vector
        self.coeffs = None                         # coefficients

    def update(self, xi, yi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        self.XTX += xi @ xi.T
        self.XTy += yi * xi.flatten()
        self.coeffs = np.linalg.inv(self.XTX + self.lambda_ridge*np.eye(self.dim)) @ self.XTy

    def predict(self, xi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        return self.coeffs @ xi.flatten()
    
    def get_coeffs(self):
        return self.coeffs
```

## LinUCB<a name=linucb></a>

LinUCB assumes that the features (i.e. $X$) and the rewards (i.e. $y$) has linear relationship for a particular arm or action. For example, in digital advertising, we may assume that a user likes a particular ad drops linearly as the age of the user increases. This is a reasonable assumption for some ads, say toy ad, since young kids like toys but their slowly lose interest as they grow older. LinUCB uses online solver for Ridge Regression to estimate the rewards. To balance exploration and exploitation, it also adds UCB in the rewards.

Focusing on disjoint arm model and consider a particular arm $a$. Following the setup of online Ridge Regression solver, the algorithm initializes the following:

$$A_a = 0$$
$$b_a = 0$$

Let $\hat{\theta_a}$ be the estimated coefficient matrix. At each round, say $t$, the feature vector $x_{t,a}$ is presented. It first performs an update of the coefficient matrix by:

$$\hat{\theta_a} = {A_a}^{-1} b_a$$

It then attempts to estimate the reward for the feature vector $x_{t,a}$ with the inclusion of upper confidence bound (UCB). Let $p_{t,a}$ be the estimated reward for arm $a$ at round $t$, we have:

$$p_{t,a} = \hat{\theta_a}^\top x_{t,a} + \alpha \sqrt{x_{t,a}^\top {A_a}^{-1} x_{t,a}}$$

where $\alpha$ controls the exploration. With the estimated reward, LinUCB may decide whether to choose this arm. Comparing rewards of all arms, it chooses the arm that gives the highest estimated reward.

If this arm is chosen, the action is executed, and the actual reward $r_{t,a}$ will be observed. Given the actual reward, LinUCB further updates the following quantities which will refine $\hat{\theta_a}$:

$$A_a \leftarrow A_a + x_{t,a} x_{t,a}^\top$$
$$b_a \leftarrow b_a + r_{t} x_{t,a}$$

The process continues until the algorithm terminates.

## A Toy Example<a name=example></a>

We shall reuse our digital advertising as an example to illustrate LinUCB. Here, we assume that there are five user age groups and four types of advertisements: **Toys**, **Cars**, **Sports** and **Foods**.

The following table defines environment or the **true probability** that a user with a given feature will pick a particular ad:

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

![linucb-ground-truth](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/linucb/fig-linucb-ground-truth-vs-prediction.svg)

In the plot, the solid lines show the ground truth and the dotted lines show the linear function fitted by LinUCB. As can be seen, while none of the arms behaves exactly linearly to the age, most of them can be fitted into a linear function to capture their trend except for `Sports`. For `Sports`, users in both young age and old age have less interest but users in middle age have very strong interest. This has created a bell-shape like curve which cannot be described by a linear function. Later, we shall see the impact of this user behaviour on LinUCB performance.

## Implementation<a name=codes></a>

Here we focus on the disjoint model presented by by Li *et al.* in their work [here](https://arxiv.org/pdf/1003.0146.pdf). Implementation of LinUCB requires online Ridge Regression. In the disjoint model, arms are assumed to be independent, thus we can managed them independently as separate instances. The following is the implementation:

```python
class OnlineRidgeRegression:

    def __init__(self, num_features, lambda_ridge, alpha=1.0):
        self.lambda_ridge = lambda_ridge
        self.dim = num_features+1     # dimension of the problem
        self.A = np.eye(self.dim)     # initialize X^T X matrix, or A
        self.b = np.zeros(self.dim)   # initialize X^T y vector, or b
        self.coeffs = None            # coefficients
        self.alpha = alpha            # exploration-exploitation tradeoff

    def update(self, xi, yi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        self.A += xi @ xi.T
        self.b += yi * xi.flatten()

    def predict(self, xi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        self.coeffs = np.linalg.inv(self.A) @ self.b
        pred = self.coeffs.T @ xi.flatten() + \
                self.alpha * np.sqrt(xi.T @ np.linalg.inv(self.A) @ xi.flatten())[0]
        return pred
    
    def get_coeffs(self):
        return self.coeffs

class LinUCB:
    '''
    Linear Upper Confidence Bound (LinUCB) algorithm implementing the disjoint model.
    '''

    def __init__(self, num_features, alpha=0.5):
        '''Constructor.'''
        self.num_features = num_features
        self.alpha = alpha
        self.lambda_ridge = 1.0
        self.ridge_regression = {} # to store Ridge regression of each arm
        self.all_known_arms = []

    def description(self):
        '''Return a string which describes the algorithm.'''
        return "LinUCB"

    def update_reward(self, arm, reward, context):
        '''Use this method to update the algorithm which `arm` has been
        selected under which `context, and what `reward` has been observed 
        from the environment.'''
        if arm not in self.all_known_arms: # new arm?
            self.all_known_arms.append(arm)
            self.ridge_regression[arm] = OnlineRidgeRegression(self.num_features,self.lambda_ridge)
        self.ridge_regression[arm].update(context, reward)

    def get_reward(self, arm, context):
        '''Get the reward for a particular `arm` under this `context`.'''
        if arm not in self.all_known_arms: # new arm?
            return None
        return self.ridge_regression[arm].predict(context)

    def get_best_arm(self, context):
        '''Return a tuple (arm,reward) representing the best arm and
        the corresponding average reward.'''
        if len(self.all_known_arms)==0:
            return (None,None)
        arm_reward_list = []
        for arm in self.all_known_arms:
            reward = self.ridge_regression[arm].predict(context)
            arm_reward_list.append((arm,reward))
        return max(arm_reward_list, key=lambda x: x[1])
```

## The Outcomes<a name=outcomes></a>

We setup a scenario to recommend ads to 10000 users. For each user, we pick a type and recommend 20 ads. The user will have an opportunity to click any, and we measure how many ads does the user click or like.

Our reward for each user is the click through rate (or simply click rate) which is measured by the percentage that the user has clicked out of the 20 offered ads. If the user clicked all, then the click rate is 1.0. If the user clicked into 5 of the ads to explore, the click rate is 5/20 = 0.25.

In this scenario, we also assume that we do not know the grouping of user. In other words, we only know the age of a user, but we do not know how the users are grouped. In this case, our context can be a continuous value rather than discrete. Traditional table driven Contextual MAB requires the context to be discrete.

As shown earlier when comparing the ground truth rewards (or click rates) and predicted rewards for each arm, we see that LinUCB is able to capture the trend of the user preference if the age-reward relationship is linear. LinUCB does not necessarily need to achieve high accuracy of fitting. Although the predicted rewards for each arm are as inaccurate, the objective of LinUCB is to accurately pick the arm that can produce the highest reward. In other words, for a given age value, we are only interested in which arm has the highest reward.

From the plot, we see that LinUCB performs accurately to recommend `Toys`, `Cars` and `Foods`. However, LinUCB missed `Sports`. The reason is that the age-reward curve for `Sports` is a bell-shape rather than a linear line, therefore LinUCB is unable to describe the *peak* in its reward.

In the following, we present the learning outcome of LinUCB by comparing the ground truth best ad to LinUCB's suggested best ad. We see that looping through all possible ages, while LinUCB missed all `Sports`, it only missed one for other types of ads.

```
+----------------------------------------------------------------------+
| age |   best  |  expected  | suggested |    predicted click rate     |
+     |    ad   | click rate |     ad    | [toys, cars, sports, foods] |
+-----+---------+------------------------------------------------------+
|  5  |  toys   |    0.76    |   toys    |  [0.86, 0.13, 0.39, -0.07]  |
|  6  |  toys   |    0.76    |   toys    |  [0.85, 0.14, 0.38, -0.05]  |
|  7  |  toys   |    0.76    |   toys    |  [0.84, 0.15, 0.38, -0.04]  |
|  8  |  toys   |    0.76    |   toys    |  [0.83, 0.16, 0.38, -0.02]  |
|  9  |  toys   |    0.76    |   toys    |  [0.82, 0.17, 0.38, -0.01]  |
| 10  |  toys   |    0.76    |   toys    |  [0.81, 0.18, 0.38, 0.00]   |
| 11  |  toys   |    0.76    |   toys    |  [0.80, 0.19, 0.38, 0.02]   |
| 12  |  toys   |    0.76    |   toys    |  [0.80, 0.20, 0.38, 0.03]   |
| 13  |  toys   |    0.76    |   toys    |  [0.79, 0.21, 0.37, 0.05]   |
| 14  |  toys   |    0.76    |   toys    |  [0.78, 0.22, 0.37, 0.06]   |
| 15  |  toys   |    0.76    |   toys    |  [0.77, 0.22, 0.37, 0.08]   |
| 16  |  toys   |    0.76    |   toys    |  [0.76, 0.23, 0.37, 0.09]   |
| 17  |  toys   |    0.76    |   toys    |  [0.75, 0.24, 0.37, 0.10]   |
| 18  |  toys   |    0.76    |   toys    |  [0.74, 0.25, 0.37, 0.12]   |
| 19  |  toys   |    0.76    |   toys    |  [0.73, 0.26, 0.37, 0.13]   |
| 20  |  toys   |    0.76    |   toys    |  [0.72, 0.27, 0.37, 0.15]   |
| 21  |  toys   |    0.76    |   toys    |  [0.71, 0.28, 0.36, 0.16]   |
| 22  |  toys   |    0.76    |   toys    |  [0.70, 0.29, 0.36, 0.18]   |
| 23  |  toys   |    0.76    |   toys    |  [0.69, 0.30, 0.36, 0.19]   |
| 24  |  toys   |    0.76    |   toys    |  [0.69, 0.31, 0.36, 0.20]   |
| 25  |  toys   |    0.64    |   toys    |  [0.68, 0.32, 0.36, 0.22]   |
| 26  |  toys   |    0.64    |   toys    |  [0.67, 0.33, 0.36, 0.23]   |
| 27  |  toys   |    0.64    |   toys    |  [0.66, 0.34, 0.36, 0.25]   |
| 28  |  toys   |    0.64    |   toys    |  [0.65, 0.35, 0.36, 0.26]   |
| 29  |  toys   |    0.64    |   toys    |  [0.64, 0.36, 0.36, 0.28]   |
| 30  |  toys   |    0.64    |   toys    |  [0.63, 0.37, 0.36, 0.29]   |
| 31  |  toys   |    0.64    |   toys    |  [0.63, 0.38, 0.36, 0.31]   |
| 32  |  toys   |    0.64    |   toys    |  [0.62, 0.39, 0.36, 0.32]   |
| 33  |  toys   |    0.64    |   toys    |  [0.61, 0.40, 0.36, 0.33]   |
| 34  |  toys   |    0.64    |   toys    |  [0.60, 0.41, 0.36, 0.35]   |
| 35  | sports  |    0.70    |   toys    |  [0.59, 0.42, 0.36, 0.36]   |  missed
| 36  | sports  |    0.70    |   toys    |  [0.59, 0.43, 0.36, 0.38]   |  missed
| 37  | sports  |    0.70    |   toys    |  [0.58, 0.44, 0.36, 0.39]   |  missed
| 38  | sports  |    0.70    |   toys    |  [0.57, 0.45, 0.36, 0.41]   |  missed
| 39  | sports  |    0.70    |   toys    |  [0.56, 0.46, 0.36, 0.42]   |  missed
| 40  | sports  |    0.70    |   toys    |  [0.56, 0.47, 0.36, 0.43]   |  missed
| 41  | sports  |    0.70    |   toys    |  [0.55, 0.48, 0.37, 0.45]   |  missed
| 42  | sports  |    0.70    |   toys    |  [0.54, 0.49, 0.37, 0.46]   |  missed
| 43  | sports  |    0.70    |   toys    |  [0.53, 0.51, 0.37, 0.48]   |  missed
| 44  | sports  |    0.70    |   toys    |  [0.53, 0.52, 0.37, 0.49]   |  missed
| 45  |  cars   |    0.55    |   cars    |  [0.52, 0.53, 0.37, 0.51]   |
| 46  |  cars   |    0.55    |   cars    |  [0.51, 0.54, 0.37, 0.52]   |
| 47  |  cars   |    0.55    |   cars    |  [0.50, 0.55, 0.38, 0.54]   |
| 48  |  cars   |    0.55    |   cars    |  [0.50, 0.56, 0.38, 0.55]   |
| 49  |  cars   |    0.55    |   cars    |  [0.49, 0.58, 0.38, 0.57]   |
| 50  |  cars   |    0.55    |   cars    |  [0.48, 0.59, 0.38, 0.58]   |
| 51  |  cars   |    0.55    |   cars    |  [0.47, 0.60, 0.39, 0.60]   |
| 52  |  cars   |    0.55    |   cars    |  [0.47, 0.61, 0.39, 0.61]   |
| 53  |  cars   |    0.55    |   cars    |  [0.46, 0.63, 0.39, 0.63]   |
| 54  |  cars   |    0.55    |   foods   |  [0.45, 0.64, 0.40, 0.64]   |  missed
| 55  |  foods  |    0.80    |   foods   |  [0.44, 0.66, 0.40, 0.66]   |
| 56  |  foods  |    0.80    |   foods   |  [0.44, 0.67, 0.40, 0.67]   |
| 57  |  foods  |    0.80    |   foods   |  [0.43, 0.68, 0.40, 0.69]   |
| 58  |  foods  |    0.80    |   foods   |  [0.42, 0.70, 0.41, 0.71]   |
| 59  |  foods  |    0.80    |   foods   |  [0.41, 0.71, 0.41, 0.72]   |
| 60  |  foods  |    0.80    |   foods   |  [0.41, 0.73, 0.41, 0.74]   |
| 61  |  foods  |    0.80    |   foods   |  [0.40, 0.74, 0.42, 0.76]   |
| 62  |  foods  |    0.80    |   foods   |  [0.39, 0.75, 0.42, 0.77]   |
| 63  |  foods  |    0.80    |   foods   |  [0.38, 0.77, 0.42, 0.79]   |
| 64  |  foods  |    0.80    |   foods   |  [0.38, 0.78, 0.43, 0.81]   |
| 65  |  foods  |    0.80    |   foods   |  [0.37, 0.80, 0.43, 0.82]   |
| 66  |  foods  |    0.80    |   foods   |  [0.36, 0.81, 0.43, 0.84]   |
| 67  |  foods  |    0.80    |   foods   |  [0.35, 0.83, 0.44, 0.86]   |
| 68  |  foods  |    0.80    |   foods   |  [0.35, 0.84, 0.44, 0.87]   |
| 69  |  foods  |    0.80    |   foods   |  [0.34, 0.85, 0.45, 0.89]   |
| 70  |  foods  |    0.80    |   foods   |  [0.33, 0.87, 0.45, 0.91]   |
+-----+---------+------------------------------------------------------+
```

## What's Next?<a name=next></a>

It is important to know that LinUCB requires the feature-reward relationship to be linear. In this example, we illustrate that LinUCB can be effective when the feature-reward relationship is linear, but will fail if not. In many real-world applications, the feature-reward relationship is either known to be non-linear or unknown. How do we deal with this situation?

We can consider using neural networks. A neural network is a function approximator. It can perform regression of a complex shape of curves. In the [next chapter](https://github.com/cfoh/Multi-Armed-Bandit-Example/tree/main/ffnn2/README.md), we shall illustrate how to implement a neural network for the scenario used in this chapter.
