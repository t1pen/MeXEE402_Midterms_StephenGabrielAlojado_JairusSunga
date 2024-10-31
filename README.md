<h1 align="center">Linear Regression and Logistic Regression</h1>
<p align="center"><b>Midterms for MexEE 402 - Electives 2
<br> Prepared by: Stephen Gabriel S. Alojado and Jairus C. Sunga</b></p>

## Introduction

In Machine Learning, regression analysis is the statistical tool that is used to model the relationship between dependent and independent variables. There are two commonly used techniques that are  designed for distinct kinds of prediction tasks those two are the Linear regression and the Logistics Regression. Predicting the continuous dependent variable from one or more independent variables is called linear regression. This method is very useful when it comes to predicting prices with the use of metrics like Mean sq. d Error (MSE) and R-squared.  In contrast, the method used for classification tasks with a categorial dependent variable is the Logistic Regression. This method is very effective with the use of classification metrics like accuracy and confusion matrices.

## Dataset Description

In this project there are two dataset that will be used which are the Car Price Prediction for Linear Regression and Customer Satisfaction for Logistic Regression. The Car Price Prediction Dataset contains various features related to automobiles, such as engine size, horsepower, fuel type, and other specifications that influence the selling price of a car. The Customer Satisfaction will focus on the feedback of the customers to predict if they are satisfied or dissatisfied. This includes the demographic profile of the customers and the service rating as well as the purchase frequency. This will serve as a predictor for the target variable. With that, logistic regression will be utilized to classify customers into two categories based on their satisfaction level.

## Project Objectives

This project aims to predict the selling price of automobiles based on various independent variables. This would be able to analyze the model accuracy and interpret the influence the features such as mileage, model and other applicaple features. To assess the Customer Satisfaction Dataset to predict whether customers are satisfied or dissatisfied based on their feedback.

## Overview of Linear Regression and Logistic Regression

### 1. Linear Regression

<div align="justify">

&nbsp;&nbsp;&nbsp;&nbsp;**Linear Regression** is a **supervised learning algorithm** primarily used for predicting a **continuous dependent variable** based on one or more **independent variables**. It establishes a **linear relationship** between input features and the target output by fitting a line that minimizes **prediction errors**, capturing trends within the data. This method is especially useful in applications where **forecasting** or **trend analysis** over time is required, such as in **finance** or **real estate**.

</div>


### Key Points
- **Goal**: Predict a continuous target variable.
- **Equation**: 

<div align="center">

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

</div>

Whereas, 
  - **$y$**: predicted value  
  - **$(x_1, x_2, \dots)$**: feature variables  
  - **$(\beta_0, \beta_1, \dots)$**: coefficients

- **Loss Function**: Mean Squared Error (MSE) <br>
 <div align="center">

$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y_i})^2$$

</div>

### Figure
<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg" alt="Linear Regression" />
    <p><em>Best-fit line minimizes errors between points and the line.</em></p>
</div>

---

## 2. Logistic Regression
Logistic Regression is used for classification (e.g., Yes/No) by estimating the probability of an outcome. This is also primarily used binary calsssification problem, which is the goal is to predict if the input belongs to specific classes. Logistic Regrerssion uses sigmoid function to model the relationship between the inputs and binary outcome.

### Key Points
- **Goal**: Classify data into categories like satisfied and not satisfied based on the given factor on dataset.
- **Equation**: Uses a sigmoid function:
  $$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots)}}$$

  Whereas,
  - P(Y=1∣X) is the probability that the dependent variable 𝑌 is equal to 1 given the input features 𝑋.
  - β0 is the intercept (bias term).
  - β1, β2​, ..., βn are the coefficients (weights) for the input features X1, X2, ..., Xn.
  - e is the base of the natural logarithm, approximately equal to 2.71828.
- **Decision Boundary**: Typically $0.5$. If probability $> 0.5$, predict 1; otherwise, $0$.
- **Loss Function**: Log Loss (Cross-Entropy Loss)

  $$-\frac{1}{N} \sum_{i=1}^N \left[y_i \log(\hat{y_i}) + (1 - y_i) \log(1 - \hat{y_i})\right]$$

 Whereas,
   - N is the total number of observations.
   - yi​ is the actual label for observation 𝑖 (0 or 1).
   - pi​ is the predicted probability that the observation 𝑖 belongs to class 1.

### Example
Predicting if a patient has a disease (yes/no) based on symptoms.

### Figure
![Logistic Regression](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

*Sigmoid curve maps probabilities between 0 and 1.*

---

## Summary

<div align="center">

| Aspect               | Linear Regression                    | Logistic Regression                   |
|----------------------|--------------------------------------|---------------------------------------|
| **Use Case**         | Predict continuous values            | Predict binary categories             |
| **Output**           | Continuous values                    | Probability (0 to 1)                  |
| **Equation**         | Linear                               | Sigmoid                               |
| **Loss Function**    | Mean Squared Error                   | Log Loss                              |

</div>
