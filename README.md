<h1 align="center">Linear Regression and Logistic Regression</h1>
<p align="center"><b>Midterms for MexEE 402 - Electives 2
<br> Prepared by: Stephen Gabriel S. Alojado and Jairus C. Sunga</b></p>

## Introduction

- **Machine Learning and Regression Analysis**:
  - **Linear and Logistic Regression** are foundational techniques in machine learning, a field of artificial intelligence. These regression methods enable computers to identify patterns and make predictions based on data:
    - **Linear Regression** is utilized to predict a continuous outcome by establishing a linear relationship between independent and dependent variables. This technique is crucial for tasks where precise numerical forecasting is required, such as in finance or real estate. It fits a linear equation to the data and is evaluated using metrics like Mean Squared Error (MSE) and R-squared.
    - **Logistic Regression** is used for classification tasks, predicting categorical outcomes (like yes/no decisions). By modeling the probability of class membership, logistic regression helps in scenarios where decision-making is binary, such as in email spam detection or disease diagnosis.

### Overview of Linear Regression and Logistic Regression
---
#### 1. Linear Regression

- **Linear Regression** is a **supervised learning algorithm** primarily used for predicting a **continuous dependent variable** based on one or more **independent variables**.
- It establishes a **linear relationship** between input features and the target output by fitting a line that minimizes **prediction errors**, capturing trends within the data.


#### Key Points
- **Goal**: Predict a continuous target variable.
- **Equation**: 

<div align="center">

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

</div>

Whereas, 
  - **$y$**: predicted value  
  - **$(x_1, x_2, \dots)$**: feature variables  
  - **$(\beta_0, \beta_1, \dots)$**: coefficients


#### Figure
<div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg" alt="Linear Regression" />
    <p><em>Best-fit line minimizes errors between points and the line.</em></p>
</div>

---

#### 2. Logistic Regression
- **Logistic Regression** is used for **classification** (e.g., Yes/No) by estimating the probability of an outcome.
- This method is primarily used for **binary classification** problems, where the goal is to predict if the input belongs to specific classes.
- **Logistic Regression** uses the **sigmoid function** to model the relationship between the inputs and the binary outcome.

#### Key Points
- **Goal**: Classify data into categories like satisfied and not satisfied based on the given factor on dataset.
- **Equation**: $$\ln \left(\frac{p}{1-p}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \beta_3X_3 + \beta_4X_4$$

Whereas,
- $p$: The probability of the dependent event occurring.
- $β₀, β₁, β₂, β₃, β₄$ (Beta coefficients): 
  - $β₀$ (Beta_0): The intercept or constant term in the regression equation, representing the log-odds of the outcome when all predictors are held at zero.
  - $β₁, β₂, β₃, β₄$: Coefficients for predictors $( X_1, X_2, X_3, )$ and $( X_4 )$ respectively. Each coefficient quantifies the change in the log-odds of the outcome per unit change in the corresponding predictor, assuming other variables are held constant.
- $X₁, X₂, X₃, X₄$: Predictor variables or features that influence the response variable.

#### Figure
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png" alt="Logistic Regression">
</p>
<p align="center">
  <i>Sigmoid curve maps probabilities between 0 and 1.</i>
</p>

---

### Summary

<div align="center">

| Aspect               | Linear Regression                    | Logistic Regression                   |
|----------------------|--------------------------------------|---------------------------------------|
| **Use Case**         | Predict continuous values            | Predict binary categories             |
| **Output**           | Continuous values                    | Probability (0 to 1)                  |

</div>

## Project Objectives
- This is a pair project for **MexEE 402 - Electives 2**, where we are tasked with analyzing and create a model for two datasets, one for Linear Regression and one for Logistic Regression.
- In this project, we will be able to execute processes on creating a model for regression analysis such as **understanding the data**, **executing exploratory data analysis**, **data pre-processing**, **creating training and test sets, building a model**, and **finally assess the model through using evaluation metrics.**
- Present the project and explain the process that was taken to build the model for regression analysis.
- Interpret the result derived from the regression analysis. 


## About the Dataset

### 1. Car Price Prediction

<p align="center">
  <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/car_image.png?raw=true" alt="Car Manufacturing Plant" width="400">
</p>

- **Dataset Context**: This dataset was gathered by a consulting firm to assist Geely Auto, a Chinese automobile company aspiring to enter the US market. The data will help understand factors influencing car prices in America, which may differ from the Chinese market.
  
- **Problem Statement**: The goal is to identify significant variables that predict the price of cars and understand how these variables impact car pricing. This understanding will aid Geely Auto in designing cars and forming business strategies tailored for the US market.
  
- **Business Goal**: By modeling car prices using multiple linear regression analysis based on various independent variables, Geely Auto aims to grasp the pricing dynamics in the new market to effectively compete with US and European car manufacturers.
  
- **Dataset Content**: The dataset includes a wide range of car attributes across different types and categories prevalent in the American market, offering insights into the competitive landscape and consumer preferences.


- **About the Datasets**:
  - **Linear Regression Dataset**: "Car Price Prediction" aims to predict the pricing of cars based on various attributes like make, model, mileage, etc.
  - **Logistic Regression Dataset**: "Customer Satisfaction" focuses on predicting whether a customer is satisfied or not, based on factors related to service and product quality.

In this project there are two dataset that will be used which are the Car Price Prediction for Linear Regression and Customer Satisfaction for Logistic Regression. The Car Price Prediction Dataset contains various features related to automobiles, such as engine size, horsepower, fuel type, and other specifications that influence the selling price of a car. The Customer Satisfaction will focus on the feedback of the customers to predict if they are satisfied or dissatisfied. This includes the demographic profile of the customers and the service rating as well as the purchase frequency. This will serve as a predictor for the target variable. With that, logistic regression will be utilized to classify customers into two categories based on their satisfaction level.




