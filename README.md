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

This pair project for **MexEE 402 - Electives 2** aims:

- To analyze and create a model for two datasets, one for Linear Regression and one for Logistic Regression.
- To integrate the use of Python Programming along with other libraries in creating and building the model.
- To execute processes on creating a model for regression analysis such as *understanding the data*, *executing exploratory data analysis*, *data pre-processing*, *creating training and test sets, building a model*, and *finally assess the model through using evaluation metrics.*
- To present the project and explain the process that was taken to build the model for regression analysis.
- To interpret the result derived from the regression analysis. 


## About the Dataset

### 1. Car Price Prediction

<p align="center">
  <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/car_image.png?raw=true" alt="Car Manufacturing Plant" width="400">
</p>

#### Description

- This dataset was gathered by a consulting firm to assist Geely Auto, a Chinese automobile company aspiring to enter the US market. The data will help understand factors influencing car prices in America, which may differ from the Chinese market.
  
- The goal is to identify significant variables that predict the price of cars *(dependent variable)* and understand how these variables impact car pricing. This understanding will aid Geely Auto in designing cars and forming business strategies tailored for the US market.
  
- By modeling car prices using *multiple linear regression analysis* based on various independent variables, Geely Auto aims to grasp the pricing dynamics in the new market to effectively compete with US and European car manufacturers.
  
- The dataset includes a wide range of car attributes across different types and categories prevalent in the American market, offering insights into the competitive landscape and consumer preferences.

#### Key Features

- **Categorical Data**

  - `symboling` - Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.

  - `CarName` - Name of car *(company and model)*

  - `fueltype`- Car fuel type *(gas or diesel)*
  
  - `aspiration` - Aspiration used in a car *(std, turbo)*
  
  - `doornumber` - Number of doors in a car *(two, four)*		
  
  - `carbody`	- type of body of car	*(sedan, hatchback, etc.)*
  
  - `drivewheel`	- type of drive wheel	*(fwd, rwd, 4wd)*
  
  - `enginelocation` - Location of car engine	*(front, rear)*
  
  - `enginetype` - Type of engine *(dohc, ohc, etc.)*
  
  - `cylindernumber` - cylinder placed in the car	*(four, three, etc.)*		
  
  - `fuelsystem` - Fuel system of car *(mpfi, 2bbl, etc.)* 

- **Continuous Data**

  - `Car_ID` - Unique id of each observation (Interger)		
  
  - `wheelbase` - Weelbase of car 		
  
  - `carlength` - Length of car 		
  
  - `carwidth` - Width of car 		
  
  - `carheight`	- height of car 		
  
  - `curbweight`- The weight of a car without occupants or baggage. 		
  
  - `enginesize` - Size of car engine 		

  - `boreratio` - Boreratio of car 		
  
  - `stroke` - Stroke or volume inside the engine 		
  
  - `compressionratio` - compression ratio of car 		
  
  - `horsepower` - Horsepower 		
  
  - `peakrpm` - car peak rpm 		
  
  - `citympg` - Mileage in city 		
  
  - `highwaympg` - Mileage on highway 		
  
  - `price` *(Dependent variable)* - Price of car

### 2. Customer Satisfaction at a Kashmir Cafe

<p align="center">
  <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/SATISFACTION.png?raw=true" alt="Customer Satisfaction" width="400">
</p>

#### Description

- The dataset originates from a customer satisfaction survey at Kashmir Cafe, designed to understand and improve customer experiences. It includes detailed ratings for *overall delivery experience*, *food quality*, and *speed of delivery*, along with binary feedback on *order accuracy*.

- Each customer is uniquely identified by a Customer ID. The *Overall Delivery Experience*, *Food Quality*, *Speed of Delivery* is rated on a scale from 1 (very dissatisfied) to 5 (very satisfied).

- *Order Accuracy* is recorded as either 'Yes' or 'No,' and is used as the dependent variable for logistic regression to predict the accuracy of orders based on other factors in the dataset.

- It is particularly used for logistic regression analysis to predict order accuracy, aiding in understanding factors that influence whether orders are correctly fulfilled.

#### Key Features

- **Categorical Data**

  - `order_accuracy` - completenes of the order *(Yes or No)*

- **Continuous Data**

  - `delivery_experience` - delivery experience rating *(Scale of 1 - 5)*

  - `food_quality` - quality of the food *(Scale of 1 - 5)*
  
  - `delivery_speed` - speed of the delivery *(Scale of 1 -5)*

## Car Price Prediction Linear Regression Model
- In this section, we will discuss about the process taken by the pair to analyze, and build a linear regression model for the given dataset for predicting the Car Price.

### 1. Preparing the libraries
- For use to be able to easily analyze the dataset given, we will be needing various `libraries` for our program. This involves the libraries meant for interfacing the dataset, data visualization, building the model, and evaluation the model. The following code for importing the libraries is shown below.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
- Libraries used was primarily, `pandas`, `seaborn`, `numpy`, and `sklearn`. 


### 2. Data Understanding
- First we need to load our dataset and show the first rows of the dataset using the code below.

``` python
cars = pd.read_csv('CarPrice_Assignment.csv')
cars.head()
```
- Given the data, it involves various data types describing the specification of the car. With this we still need to understand the given dataset through programming techniques such as `.info()`, `.describe()`, and `.head()`.

- Using `.info()`, we have been given the datatypes of each column and the shape of our dataset.

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 205 entries, 0 to 204
Data columns (total 26 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   car_ID            205 non-null    int64  
 1   symboling         205 non-null    int64  
 2   CarName           205 non-null    object 
 3   fueltype          205 non-null    object 
 4   aspiration        205 non-null    object 
 5   doornumber        205 non-null    object 
 6   carbody           205 non-null    object 
 7   drivewheel        205 non-null    object 
 8   enginelocation    205 non-null    object 
 9   wheelbase         205 non-null    float64
 10  carlength         205 non-null    float64
...
 24  highwaympg        205 non-null    int64  
 25  price             205 non-null    float64
dtypes: float64(8), int64(8), object(10)
```

- Based on the information above, we have *26 columns* and *205 rows* of data. It is also determined that we have `object` type data (mainly categorical) so we need to process it on the later part.

- For the feature `symboling` it is stated that it is an `int64` datatype. However, it is stated that it is a categorical data so we make it as an `object` type of data (but later on will not be used due to being a weak predictor).

### 3. Data Preprocessing
  - Convert "symboling" to Categorical Data

    - After referring to the data dictionary, we determined that the `symboling` feature is a categorical variable. Therefore, we need to convert it from its integer representation to an object type. This allows the model to treat it as a category rather than a numerical value, which is important for proper analysis and interpretation.
      
```python
cars['symboling'] = cars['symboling'].astype('object')
```

- In using the code below we will identify the continuous and categorical features in our dataset.

```python
categorical = cars.select_dtypes(include=['object']).columns
continuous = cars.select_dtypes(include=['float64', 'int64']).columns

print(f"Number of categorical features: {len(categorical)}")
print(f"Number of continuous features: {len(continuous)}")
```
- This results to:
  - Number of categorical features: 11
  - Number of continuous features: 15

 - We can print out the names of the categorical and continuous features identified in the previous step.

 ```python
 Categorical Features:
Index(['symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'enginetype',
       'cylindernumber', 'fuelsystem'],
      dtype='object')

Continuous Features:
Index(['car_ID', 'wheelbase', 'carlength', 'carwidth', 'carheight',
       'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price'],
      dtype='object')
```

- Checking for NaN Values in the Dataset
  - Before proceeding with data analysis and model building, it's important to check for any missing values (NaN) in the dataset.
    -The following code snippet checks for NaN values in the dataset:
    ```python
    cars.isnull().sum()
    ```

```python
car_ID              0
symboling           0
CarName             0
fueltype            0
aspiration          0
doornumber          0
carbody             0
drivewheel          0
enginelocation      0
wheelbase           0
carlength           0
carwidth            0
carheight           0
curbweight          0
enginetype          0
cylindernumber      0
enginesize          0
fuelsystem          0
boreratio           0
stroke              0
compressionratio    0
horsepower          0
peakrpm             0
citympg             0
highwaympg          0
price               0
dtype: int64
```

  - The output indicates that there are no missing values in any of the columns

### 3. Data Cleaning
- Data cleaning is an essential step to correct errors in our dataset and prepare it for data visualization and exploratory data analysis (EDA). This process ensures that our data is accurate and reliable for the analysis.

- To analyze the dataset effectively, we first separate the company name and car model from the `CarName` feature. The company name will be extracted using the following code:

```python
cars['CompanyName'] = cars['CarName'].apply(lambda x: x.split(" ")[0])
```

- Once we have the `CompanyName`, we can drop the original `CarName` column in favor of the new feature. We can accomplish this with the following code:

```python
cars.drop(columns='CarName', inplace=True)
```

- Fixing Misspelled Company Names
    -There's need to correct some misspelled company names in the dataset. The entries that require fixing are:

- maxda → mazda
- Nissan → nissan
- porcshce → porsche
- toyouta → toyota
- vokswagen & vw → volkswagen

```python
cars['CompanyName'].replace({
    'maxda': 'mazda',
    'Nissan': 'nissan',
    'porcshce': 'porsche',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen'
}, inplace=True)
```
```python
array(['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'buick', 'mercury', 'mitsubishi',
       'nissan', 'peugeot', 'plymouth', 'porsche', 'renault', 'saab',
       'subaru', 'toyota', 'volkswagen', 'volvo'], dtype=object)
```

### 4. Data Visualization
- This is a necessary step due to the data having some entries that needs to be processed. Also, we need to analyze the given dataset so we can evaluate what specific variables correlates to our dependent variable (price).
    - In the first subplot, the histogram provides a visual representation of the distribution of car prices, with the Kernel Density Estimate (KDE) curve overlaying the histogram for better clarity of the price distribution shape.
    - The second subplot, the box plot, allows us to see the median, quartiles, and any outliers in the price data, giving us further insights into the pricing distribution.

![image](https://github.com/user-attachments/assets/e88a6b4d-429e-41fe-acf4-27cd2f04530c)

### Visualizing Categorical Data

- Each illustrating the number of cars within different categories of the specified features.

  ![image](https://github.com/user-attachments/assets/d69f9548-86ec-4b3e-bd2e-23140258ca2a)
  ![image](https://github.com/user-attachments/assets/29345b57-3071-4c9f-8830-af8b8184eb1c)

- Box plots provide a visual summary of the central tendency, variability, and outliers of the car prices across different categorical features.

  ![image](https://github.com/user-attachments/assets/46f20145-ed38-407a-8d99-725b4199805a)
  ![image](https://github.com/user-attachments/assets/98746dfc-e48e-484d-8b49-10969960d21a)

- Bar plots effectively illustrate the average car prices across different categories, allowing us to identify which features are associated with higher or lower prices.

  ![image](https://github.com/user-attachments/assets/7f28c90c-9478-46af-ad51-46327943c1f2)
  ![image](https://github.com/user-attachments/assets/b30174c3-18a0-4820-bd3a-1625c18645f0)

 - After visualization of the categorical features here are some of the insights:

   - CompanyName: There is a noticeable difference in mean prices across different car manufacturers, which suggests that CompanyName could be a useful feature, although it may need to be encoded appropriately.

   - doornumber: The difference in mean price between cars with two and four doors is relatively small. This feature may have low predictive power for price.

    - fueltype: There is a noticeable price difference between diesel and gas cars. fueltype could be useful.

    - aspiration: There is a difference in mean price between turbo and standard aspiration, so aspiration could be a useful feature.

    - carbody: Different body styles show distinct mean prices, indicating that carbody may contribute to price prediction.

    - enginelocation: A significant price difference exists between front and rear engine locations. enginelocation might be an important feature.

    - drivewheel: The mean prices vary by drive type (e.g., rwd, fwd, 4wd), so drivewheel could be a useful feature.

    - enginetype: There are noticeable price variations across different engine types, so enginetype might also be a useful predictor.

    - cylindernumber: The mean price varies with the number of cylinders, suggesting cylindernumber could be valuable.

    - fuelsystem: While there is some variation in mean price across different fuel systems, it is relatively minor, indicating fuelsystem might be less significant.

- To understand the relationship between the continuous features and the target variable (price), we can use scatter plots along with regression lines.

![image](https://github.com/user-attachments/assets/de38f18f-d7c3-42c7-b5c5-44f6c6ce4850)
![image](https://github.com/user-attachments/assets/0d45310c-0cfd-4447-bb5d-f1a56240868a)

- To gain insights into the relationships between the numerical features and the target variable (price), we will compute the correlation matrix. This matrix reveals how strongly the features correlate with each other and with the target variable, helping us identify which features may have a significant impact on car prices.
  
  ![image](https://github.com/user-attachments/assets/0b884750-8c26-4462-bc81-a4abd8b73219)

- In this heatmap, the correlation coefficients are displayed, with values ranging from -1 to 1. A value close to 1 indicates a strong positive correlation, meaning that as one variable increases, the other tends to increase as well. Conversely, a value close to -1 indicates a strong negative correlation. Values around 0 suggest little to no linear relationship between the variables.

- After visualizing the continuous features, here are some insights:

  - curbweight (correlation ~0.84): This has a strong positive correlation with price, indicating that heavier cars tend to be more expensive.
  - enginesize (correlation ~0.87): This feature shows a very strong positive correlation with price, making it an essential predictor.
  - boreratio (correlation ~0.55): This feature has a moderate positive correlation with price, suggesting that cars with a higher bore ratio
  - horsepower (correlation ~0.81): Higher horsepower is associated with a higher price, suggesting it is an influential predictor.
  - carlength (correlation ~0.68): This feature also has a moderate positive correlation with price, suggesting that longer cars are generally more expensive.
  - carwidth (correlation ~0.76): This feature also has a strong positive correlation with price, suggesting that wider cars are generally more expensive.
  - wheelbase (correlation ~0.58): Moderate positive correlation, which could still contribute useful information to the model.
  - citympg and highwaympg (correlations ~-0.7): Both features have a strong negative correlation with price, indicating that cars with higher fuel efficiency tend to be less expensive.
  - 
- After visualization here are the features to be retained for building the model:

- Categorical Features
  - `CompanyName`
  - `fueltype`
  - `aspiration`
  - `carbody`
  - `enginelocation`
  - `drivewheel`
  - `enginetype`
  - `cylindernumber`

- Continuous Features
  - `curbweight`
  - `enginesize`
  - `boreratio`
  - `horsepower`
  - `carlength`
  - `carwidth`
  - `wheelbase`
  - `citympg`
  - `highwaympg`

### 5. Feature Engineering
- In this section, we will convert categorical data to numerical formats using One-Hot Encoding and Label Encoding. Additionally, we will drop unnecessary features based on the insights derived from the Exploratory Data Analysis (EDA).

  - By using this code `cars[con_sig].head()`, we can display the first few rows of the continuous features in the dataset. This allows us to visually inspect the values and understand the distribution and range of features.
    
![image](https://github.com/user-attachments/assets/7105dd75-b962-426e-a330-2296be74dc28)

### Dropping unnecessary features



### Categorical Feature Engineering

- Converting cylindernumber from object datatype to numerical.
  


- Converting other categorical features to numerical using One Hot Encoding



- Moving price to the rightmost column
  


- Using this code cars_final.info(), we can retrieve a concise summary of the cars_final DataFrame, which includes essential information about the dataset's structure and content.

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 205 entries, 0 to 204
Data columns (total 54 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   cylindernumber           205 non-null    int64  
 1   wheelbase                205 non-null    float64
 2   carlength                205 non-null    float64
 3   carwidth                 205 non-null    float64
 4   curbweight               205 non-null    int64  
 5   enginesize               205 non-null    int64  
 6   boreratio                205 non-null    float64
 7   horsepower               205 non-null    int64  
 8   citympg                  205 non-null    int64  
 9   highwaympg               205 non-null    int64  
 10  CompanyName_alfa-romero  205 non-null    int64  
 11  CompanyName_audi         205 non-null    int64  
 12  CompanyName_bmw          205 non-null    int64  
 13  CompanyName_buick        205 non-null    int64  
 14  CompanyName_chevrolet    205 non-null    int64  
 15  CompanyName_dodge        205 non-null    int64  
 16  CompanyName_honda        205 non-null    int64  
 17  CompanyName_isuzu        205 non-null    int64  
 18  CompanyName_jaguar       205 non-null    int64  
 19  CompanyName_mazda        205 non-null    int64  
 20  CompanyName_mercury      205 non-null    int64  
 21  CompanyName_mitsubishi   205 non-null    int64  
 22  CompanyName_nissan       205 non-null    int64  
 23  CompanyName_peugeot      205 non-null    int64  
 24  CompanyName_plymouth     205 non-null    int64  
 25  CompanyName_porsche      205 non-null    int64  
 26  CompanyName_renault      205 non-null    int64  
 27  CompanyName_saab         205 non-null    int64  
 28  CompanyName_subaru       205 non-null    int64  
 29  CompanyName_toyota       205 non-null    int64  
 30  CompanyName_volkswagen   205 non-null    int64  
 31  CompanyName_volvo        205 non-null    int64  
 32  fueltype_diesel          205 non-null    int64  
 33  fueltype_gas             205 non-null    int64  
 34  aspiration_std           205 non-null    int64  
 35  aspiration_turbo         205 non-null    int64  
 36  carbody_convertible      205 non-null    int64  
 37  carbody_hardtop          205 non-null    int64  
 38  carbody_hatchback        205 non-null    int64  
 39  carbody_sedan            205 non-null    int64  
 40  carbody_wagon            205 non-null    int64  
 41  enginelocation_front     205 non-null    int64  
 42  enginelocation_rear      205 non-null    int64  
 43  drivewheel_4wd           205 non-null    int64  
 44  drivewheel_fwd           205 non-null    int64  
 45  drivewheel_rwd           205 non-null    int64  
 46  enginetype_dohc          205 non-null    int64  
 47  enginetype_dohcv         205 non-null    int64  
 48  enginetype_l             205 non-null    int64  
 49  enginetype_ohc           205 non-null    int64  
 50  enginetype_ohcf          205 non-null    int64  
 51  enginetype_ohcv          205 non-null    int64  
 52  enginetype_rotor         205 non-null    int64  
 53  price                    205 non-null    float64
dtypes: float64(5), int64(49)
memory usage: 86.6 KB
```

### 6. Building the Model
- This is the section where we will build our Linear Regression Model.

- Getting X and y Variables
  - To separate our features and target variable, we will create our independent variable matrix 𝑋 and dependent variable vector 𝑦 as follows:

```python
X = cars_final.iloc[:,:-1].values
pd.DataFrame(X)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>111.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>88.6</td>
      <td>168.8</td>
      <td>64.1</td>
      <td>2548.0</td>
      <td>130.0</td>
      <td>3.47</td>
      <td>111.0</td>
      <td>21.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.0</td>
      <td>94.5</td>
      <td>171.2</td>
      <td>65.5</td>
      <td>2823.0</td>
      <td>152.0</td>
      <td>2.68</td>
      <td>154.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>99.8</td>
      <td>176.6</td>
      <td>66.2</td>
      <td>2337.0</td>
      <td>109.0</td>
      <td>3.19</td>
      <td>102.0</td>
      <td>24.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>99.4</td>
      <td>176.6</td>
      <td>66.4</td>
      <td>2824.0</td>
      <td>136.0</td>
      <td>3.19</td>
      <td>115.0</td>
      <td>18.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>4.0</td>
      <td>109.1</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>2952.0</td>
      <td>141.0</td>
      <td>3.78</td>
      <td>114.0</td>
      <td>23.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>4.0</td>
      <td>109.1</td>
      <td>188.8</td>
      <td>68.8</td>
      <td>3049.0</td>
      <td>141.0</td>
      <td>3.78</td>
      <td>160.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>6.0</td>
      <td>109.1</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>3012.0</td>
      <td>173.0</td>
      <td>3.58</td>
      <td>134.0</td>
      <td>18.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.0</td>
      <td>109.1</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>3217.0</td>
      <td>145.0</td>
      <td>3.01</td>
      <td>106.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>4.0</td>
      <td>109.1</td>
      <td>188.8</td>
      <td>68.9</td>
      <td>3062.0</td>
      <td>141.0</td>
      <td>3.78</td>
      <td>114.0</td>
      <td>19.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 53 columns</p>
</div>

```python
y = cars_final.iloc[:,-1].values
pd.DataFrame(y)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13495.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13950.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17450.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>16845.0</td>
    </tr>
    <tr>
      <th>201</th>
      <td>19045.0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>21485.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>22470.0</td>
    </tr>
    <tr>
      <th>204</th>
      <td>22625.0</td>
    </tr>
  </tbody>
</table>
<p>205 rows × 1 columns</p>
</div>

 
## Customer Satisfaction Analysis Logistic Regression Model
- In this section, we will discuss about the process taken by the pair to analyze, and build a linear regression model for the given dataset for predicting the Car Price.

### 1. Importing Required Libraries
- In this section, we will import the necessary libraries and modules that are essential for performing data analysis, visualization, and building the logistic regression model for our customer satisfaction analysis. These libraries provide the tools needed for data manipulation, statistical modeling, and visual representation of our findings.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
```
### 2. Loading Dataset
- This process involves importing customer survey data from a CSV file into a DataFrame for subsequent analysis and visualization. This displays the some rows to provide a quick overview of the data.
  
``` python
dataset = pd.read_csv('Customer-survey-data.csv')
dataset.head()
```
Verifying Dataset Dimensions

-This step checks the number of rows and columns in the dataset, providing an overview of its size and structure to help gauge the amount of data available for analysis.
``` python
rows, columns = dataset.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
```
-In the data there's a lengthy column names and it is not visually appealing with that `Renaming Dataset Columns` would be great for improved clarity and readability, making the data easier to work with and understand at a glance.

  - Previous data with lengthy names
![image](https://github.com/user-attachments/assets/087c1903-62d5-43fc-a9ea-cf2972e040fa)

``` python
dataset = dataset.rename(columns={'Customer': 'Customer',
       'How satisfied were you with your overall delivery experience at Ali?                    1-5 where 1 = extremely dissatisfied and 5 = extremely satisfied': 'delivery_experience',
       'How satisfied were you with the quality of the food at Alis?                             1-5 where 1 = extremely dissatisfied and 5 = extremely satisfied': 'food_quality',
       'How satisfied were you with the speed of delivery at Alis?                                1-5 where 1 = extremely dissatisfied and 5 = extremely satisfied': 'delivery_speed',
       'Was your order accurate? Please respond yes or no.': 'Order_Accuracy'})
dataset
```
  - After `Renaming`

<p align="center">
  <img src="https://github.com/user-attachments/assets/8c7ab110-c6a5-4e8c-be05-75a35c731a9d" alt="Description of Image" />
</p>

-In `.info()`, this will explore data types, non-null counts, and memory usage, which gives insight into the structure and quality of the data, identifying any missing or incorrect values that might require preprocessing.

``` python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10616 entries, 0 to 10615
Data columns (total 5 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   Customer             10616 non-null  int64  
 1   delivery_experience  10198 non-null  float64
 2   food_quality         10364 non-null  float64
 3   delivery_speed       10377 non-null  float64
 4   Order_Accuracy       9956 non-null   object 
dtypes: float64(3), int64(1), object(1)
```

