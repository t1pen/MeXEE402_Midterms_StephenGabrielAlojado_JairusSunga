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
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg" alt="Linear Regression", width=500 />
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
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png" alt="Logistic Regression", width=500>
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
  - **Number of categorical features:** 11
  - **Number of continuous features:** 15

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
  - There's need to correct some misspelled company names in the dataset. The entries that require fixing are:
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
- After processing the mispelled names, `CompanyName` will have correct entries.

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

  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_Price%20Dist.png?raw=true" alt="PriceDist">
  </p>


#### Visualizing Categorical Data

- Each illustrating the number of cars within different categories of the specified features.

  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_Countplot.png?raw=true" alt="CountPlot">
  </p>

- Box plots provide a visual summary of the central tendency, variability, and outliers of the car prices across different categorical features.

  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_BoxPlot.png?raw=true" alt="BoxPlot">
  </p>

- Bar plots effectively illustrate the average car prices across different categories, allowing us to identify which features are associated with higher or lower prices.

  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_MeanPrice.png?raw=true" alt="MeanPrice">
  </p>


 - After visualization of the categorical features here are some of the insights:

   - `CompanyName`: There is a noticeable difference in mean prices across different car manufacturers, which suggests that CompanyName could be a useful feature, although it may need to be encoded appropriately.

   - `doornumber`: The difference in mean price between cars with two and four doors is relatively small. This feature may have low predictive power for price.

    - `fueltype`: There is a noticeable price difference between diesel and gas cars. fueltype could be useful.

    - `aspiration`: There is a difference in mean price between turbo and standard aspiration, so aspiration could be a useful feature.

    - `carbody`: Different body styles show distinct mean prices, indicating that carbody may contribute to price prediction.

    - `enginelocation`: A significant price difference exists between front and rear engine locations. enginelocation might be an important feature.

    - `drivewheel`: The mean prices vary by drive type (e.g., rwd, fwd, 4wd), so drivewheel could be a useful feature.

    - `enginetype`: There are noticeable price variations across different engine types, so enginetype might also be a useful predictor.

    - `cylindernumber`: The mean price varies with the number of cylinders, suggesting cylindernumber could be valuable.

    - `fuelsystem`: While there is some variation in mean price across different fuel systems, it is relatively minor, indicating fuelsystem might be less significant.

- To understand the relationship between the continuous features and the target variable (price), we can use scatter plots along with regression lines.

  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_scatterplot.png?raw=true" alt="ScatterPlot">
  </p>

- To gain insights into the relationships between the numerical features and the target variable (price), we will compute the correlation matrix. This matrix reveals how strongly the features correlate with each other and with the target variable, helping us identify which features may have a significant impact on car prices.
  
  <p align="center">
    <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_Corr.png?raw=true" alt="CorrMat">
  </p>

- In this heatmap, the correlation coefficients are displayed, with values ranging from -1 to 1. A value close to 1 indicates a strong positive correlation, meaning that as one variable increases, the other tends to increase as well. Conversely, a value close to -1 indicates a strong negative correlation. Values around 0 suggest little to no linear relationship between the variables.

- After visualizing the continuous features, here are some insights:

  - `curbweight` *(correlation ~0.84)*: This has a strong positive correlation with price, indicating that heavier cars tend to be more expensive.
  - `enginesize` *(correlation ~0.87)*: This feature shows a very strong positive correlation with price, making it an essential predictor.
  - `boreratio` *(correlation ~0.55)*: This feature has a moderate positive correlation with price, suggesting that cars with a higher bore ratio
  - `horsepower` *(correlation ~0.81)*: Higher horsepower is associated with a higher price, suggesting it is an influential predictor.
  - `carlength` *(correlation ~0.68)*: This feature also has a moderate positive correlation with price, suggesting that longer cars are generally more expensive.
  - `carwidth` *(correlation ~0.76)*: This feature also has a strong positive correlation with price, suggesting that wider cars are generally more expensive.
  - `wheelbase` *(correlation ~0.58)*: Moderate positive correlation, which could still contribute useful information to the model.
  - `citympg` and `highwaympg` *(correlations ~-0.7)*: Both features have a strong negative correlation with price, indicating that cars with higher fuel efficiency tend to be less expensive.
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

#### Viewing and Inspecting our Significant Features
- In this section, we will convert categorical data to numerical formats using One-Hot Encoding and Label Encoding. Additionally, we will drop unnecessary features based on the insights derived from the Exploratory Data Analysis (EDA).

  - By using this code `cars[con_sig].head()`, we can display the first few rows of the continuous features in the dataset. This allows us to visually inspect the values and understand the distribution and range of features.

    <p align="center">
      <img src="https://github.com/user-attachments/assets/7105dd75-b962-426e-a330-2296be74dc28" alt="ConSig">
    </p>

  - By using this code `cars[cat_sig].head()`, we can display the first few rows of the categorical features in the dataset.

    <p align="center">
      <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_CatSig.png?raw=true" alt="CorrMat">
    </p>


#### Categorical Feature Engineering

- Converting cylindernumber from object datatype to numerical.
  
  ```python
  cars['cylindernumber'] = cars['cylindernumber'].replace({'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12}).astype('int64')
  cars[cat_sig].head()
  ```

- Converting other categorical features to numerical using One Hot Encoding
  ```python
  cat_ohe = ['CompanyName', 'fueltype', 'aspiration', 'carbody', 'enginelocation', 'drivewheel', 'enginetype']

  cars_final = pd.get_dummies(cars, columns=cat_ohe, dtype='int64')

  cars_final.head()
  ```

- Moving price to the rightmost column
  ``` python
  cols = list(cars_final.columns)
  cols.remove('price')
  cols.append('price')

  cars_final = cars_final[cols]
  ```


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
  ...  
  51  enginetype_ohcv          205 non-null    int64  
  52  enginetype_rotor         205 non-null    int64  
  53  price                    205 non-null    float64
  dtypes: float64(5), int64(49)
  memory usage: 86.6 KB
  ```
- We now have finally processed our dataset for building our linear regression model.

### 6. Building the Model
- This is the section where we will build our Linear Regression Model.

- Getting X and y Variables
  - To separate our features and target variable, we will create our independent variable matrix 𝑋 and dependent variable vector 𝑦 as follows:

    ```python
    X = cars_final.iloc[:,:-1].values
    pd.DataFrame(X)
    ```

    ```python
    y = cars_final.iloc[:,-1].values
    pd.DataFrame(y)
    ```

#### Train test split

- Splitting our dataset by `80-20` for our model training and model testing.

  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
  print("X_train shape:", X_train.shape)
  print("X_test shape:", X_test.shape)
  print("y_train shape:", y_train.shape)
  print("y_test shape:", y_test.shape)
  ```
- Results of the Train Test Split

  ```python
  X_train shape: (164, 53)
  X_test shape: (41, 53)
  y_train shape: (164,)
  y_test shape: (41,)
  ```

#### Training Linear Regression Model
- We use this code to train our data for Linear Regression Model.

  ```python
  lr = LinearRegression()
  lr.fit(X_train, y_train)
  ```

#### Inference

- In this section, we will make predictions using our trained Linear Regression model.

  ```python
  data_sample = np.array([[4, 88.6, 168.8, 64.1, 2548, 130, 3.47, 111, 21, 27, 
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 
                        1, 1, 0, 0, 0, 0, 0, 0]])
  predicted_price = lr.predict(data_sample)
  print(f"Predicted Price: {round(predicted_price[0],2)}")
  price_row_1 = cars_final.iloc[0]['price']
  print(f"Actual Price: {price_row_1}")
  print(f"Difference: {price_row_1 - predicted_price}")
  ```
- Note that the `data_sample` is from the dataset checking if its has predicted the price of the car base on the data entry that we have set.

  ```python
  Predicted Price: 13647.55
  Actual Price: 13495.0
  Difference: [-152.55121761]
  ```
- This show that we have a small difference from the Actual Price.

### 7. Evaluation of the Model
- To evaluate the performance of our Linear Regression model, we will compute several key metrics: *R² Score*, *Mean Absolute Error (MAE)*, *Mean Squared Error (MSE)*, and *Root Mean Squared Error (RMSE)*. This is obtained using this code

  ```python
  # Mean Absolute Error
  mae = mean_absolute_error(y_test, y_pred)
  print(f"Mean Absolute Error (MAE): {mae:.2f}")

  # Mean Squared Error
  mse = mean_squared_error(y_test, y_pred)
  print(f"Mean Squared Error (MSE): {mse:.2f}")

  # Root Mean Squared Error
  rmse = np.sqrt(mse)
  print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

  # R² Score
  r2 = r2_score(y_test, y_pred)
  print(f"R² Score: {r2:.4f}")
  ```
 
### Plotting of the Predicted vs. Actual Price

  <p align="center">
      <img src="https://github.com/t1pen/MeXEE402_Midterms_StephenGabrielAlojado_JairusSunga/blob/main/Images/Cars_ModelScatter.png?raw=true" alt="CorrMat">
  </p>

- This is the plot of the predicted price versus the actual price. It shows that there are still a difference between the two speciall at the higher prices where there is few entries.

 
## Customer Satisfaction Analysis Logistic Regression Model
- In this section, we will discuss about the process taken by the pair to analyze, and build a *logistic regression model* for the given dataset for predicting the Order Accuracy *(dependent variable)* based on the variables present in the data set.

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
#### Verifying Dataset Dimensions

- This step checks the number of rows and columns in the dataset, providing an overview of its size and structure to help gauge the amount of data available for analysis.

  ``` python
  rows, columns = dataset.shape
  print(f"Number of rows: {rows}")
  print(f"Number of columns: {columns}")
  ```
- In the data there's a lengthy column names and it is not visually appealing with that `Renaming Dataset Columns` would be great for improved clarity and readability, making the data easier to work with and understand at a glance.

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

- In `.info()`, this will explore data types, non-null counts, and memory usage, which gives insight into the structure and quality of the data, identifying any missing or incorrect values that might require preprocessing.

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


- Checking for NaN Values in the Dataset
    - To ensure the integrity of our dataset and the reliability of our logistic regression model, it is important to identify and handle any missing values. The following counts of NaN values for each column in the dataset were obtained:

      ```python
      Customer                 0
      delivery_experience    418
      food_quality           252
      delivery_speed         239
      Order_Accuracy         660
      dtype: int64
      ```

- To maintain the integrity of our dataset and enhance the performance of our logistic regression model, it is important to address the missing values identified in the previous section.

  ```python
  for col in dataset: 
      dataset[col] = dataset[col].fillna(dataset[col].median())
  ```

- Checking for NaN Values After Filling
    - After filling the missing values in our dataset, it is important to verify that all NaN values have been successfully handled

  ```python
  NaN values in dataset after handling:
  Customer               0
  delivery_experience    0
  food_quality           0
  delivery_speed         0
  Order_Accuracy         0
  dtype: int64
  ```

### 3. Data Visualization
- This is where we visualize our dataset and observe the correlation of our variables using Data Visualization Techniques

#### Visualizing Categorical Data
- This is the count plot for the `Order_Accuracy`.

  ![image](https://github.com/user-attachments/assets/f6a7b466-f0e8-4909-9638-e25be2ada36e)

#### Visualizing Continuous Data
- The continuous features we are analyzing include '`delivery_experience`', '`food_quality`', and '`delivery_speed`'. These features provide insights into customer satisfaction in terms of delivery performance and food quality. 
  
  ![image](https://github.com/user-attachments/assets/8569844a-cf57-4a26-917c-183a335c12de)

#### Visualizing Continuous Features with Box Plots
  ![image](https://github.com/user-attachments/assets/796875bd-b0b9-4b23-834b-fc78f2237ea6)

#### Visualizing Continuous Features by Order Accuracy
  ![image](https://github.com/user-attachments/assets/2e66dfd5-b7cb-41c2-9a85-ab1c0e5eb4c5)

#### Correlation Heatmap of Customer Satisfaction Variables
  - A correlation heatmap visually represents the relationships between different numerical features in the dataset. This helps identify which variables are positively or negatively correlated and to what degree.

    ![image](https://github.com/user-attachments/assets/3661f2e4-ed7b-42d1-b207-cbbe0ac4ec9e)

### 4. Building the Model
- In this section, we will build our Logistic Regression model. This involves preparing the input features and the output variable that the model will learn from.

#### Getting the Inputs and Output
- To train the model, we need to separate the input features from the output target variable. The input features consist of the numerical ratings for delivery experience, food quality, and delivery speed, while the target variable is the order accuracy

  ```python
  X = dataset.iloc[:,1:-1].values
  pd.DataFrame(X)
  ```

  ```python
  y = dataset.iloc[:,-1].values
  pd.DataFrame(y)
  ```

#### Creating the Training Set and the Test Set

- To evaluate the performance of our Logistic Regression model effectively, we need to split the dataset into training and test sets.

  ```python
  X_train shape: (8492, 3)
  X_test shape: (2124, 3)
  y_train shape: (8492,)
  y_test shape: (2124,)
  ```

#### Training the Model

- In this section, we will train our Logistic Regression model on the training dataset. We will also address the class imbalance present in the dataset by setting the `class_weight` parameter to `'balanced'`.

  ```python
  # Set class_weight to 'balanced' to address class imbalance
  model = LogisticRegression(random_state=0, class_weight='balanced')

  # Train the model on the original X_train_scaled, y_train
  model.fit(X_train, y_train)
  ```

#### Inference
- Predict `y values` based on the `X_test`.
  ```python
  y_pred = model.predict(sc.transform(X_test))
  pd.DataFrame(y_pred)
  ```

- This section will demonstrate how to make a single prediction using the trained Logistic Regression model. This example will predict the order accuracy based on provided feature values.

  ```python
  prediction = model.predict(sc.transform([[5, 5, 5]]))

  if prediction[0] == 1:
      print("Customer's Order is Complete (1)")
  elif prediction[0] == 0:
      print("Customer's Order is Incomplete (0)")
  ```

- The result shows: `Customer's Order is Complete (1)`
- The model predicted that the customer's order is complete, with a result of 1.


### 5. Evaluating the model

- Now we will evaluate our logistic regression model.

#### Accuracy

```python
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
```

- **Accuracy Score:** 0.7194
- The Logistic Regression model achieved an accuracy score of **0.7194** on the test dataset. This means that the model correctly predicted customer order outcomes approximately **71.94%** of the time.
        
#### Confusion Matrix

- The confusion matrix allows for an insightful breakdown of the model's performance and overall accuracy.

  ![image](https://github.com/user-attachments/assets/576b3bf1-6601-47f9-b51d-caafa1b46a39)

#### Summary of Evaluation
- Using the code we can summarize the evaluation.

  ```python
  accuracy = accuracy_score(y_test,y_pred)
  print(f"Accuracy Score: {accuracy:.4f}")

  print("Confusion Matrix:")
  print(cm)
  ```
- Giving this results:
  ```python
  Accuracy Score: 0.7194
  Confusion Matrix:
  [[  12  553]
  [  43 1516]]
  ```

## Results and Discussion

### Car Price Prediction (Linear Regression)

- After building the linear regression model, we are able to predict the Car Price based on the specification given. Given below is the summary of the results.

#### R² Score
- The *R² score* measures the proportion of variance in the dependent variable that can be explained by the independent variables. It ranges from 0 to 100, where a higher score indicates a better fit.
  - **R² Score:** 0.9308
  - This indicates that approximately **93.08%** of the variance in car prices is explained by our model.

#### Mean Absolute Error (MAE)
- *MAE* measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s calculated as the average over the test sample of the absolute differences between prediction and actual observation.
    - **Mean Absolute Error (MAE):** 1,686.60
    - This means, on average, our model’s predictions are off by approximately **1,686.60**.
 
#### Mean Squared Error (MSE)
- *MSE* measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. It penalizes larger errors more than smaller ones.
    - **Mean Squared Error (MSE):** 3,926,875.33
    - This indicates the overall magnitude of the error.
 
#### Root Mean Squared Error (RMSE)
- *RMSE* is the square root of the mean of the squared errors and is useful for understanding how far off predictions are from actual values.
    - **Root Mean Squared Error (RMSE):** 1,981.63
    - This means that the average prediction error is about **1,981.63**, giving us a sense of the model's accuracy in terms of actual dollar amounts.

#### Overall Discussion

- The linear regression model for predicting car prices demonstrates **strong predictive capability** as indicated by the high R² score. However, while the model appears to fit well overall, the MAE and RMSE values highlight that there are **still notable errors** in the predictions that need to be addressed. The presence of a high MSE indicates **potential issues with outliers**, which could be impacting the model’s accuracy and predictive power.

### Customer Satisfaction (Logistic Regression)

- After building the logistic regression model, we are able to predict whether the order was complete or not based on the ratings given for delivery experience, food quality, and delivery speed. Given below is the summary of the results.

#### Accuracy Score
- This means that the model correctly predicts the order accuracy approximately **71.94%** of the time.
  - **Accuracy Score:** 0.7194

- This means that it has a low score of accuracy. Based on our data analysis of the dataset, the order accuracy has little to no correlation with its predictors making our model prediction to have a low score (*Refer to* [Data Visualization](#3-data-visualization)).

#### Confusion Matrix

The confusion matrix can be interpreted as follows:

- **True Negatives (TN):** *12* - The model correctly predicted that 12 orders were accurate (i.e., the model predicted "accurate" and the actual was also "accurate").

- **False Positives (FP):** *553* - The model incorrectly predicted that 553 orders were accurate when they were not (i.e., predicted "accurate," but the actual was "inaccurate").
- **False Negatives (FN):** *43* - The model incorrectly predicted that 43 orders were inaccurate when they were actually accurate (i.e., predicted "inaccurate," but the actual was "accurate").
- **True Positives (TP):** *1516* - The model correctly predicted that 1516 orders were inaccurate.

#### Overall Discussion
- While the logistic regression model has demonstrated some ability to predict order accuracy, its performance metrics indicate significant room for improvement. The high false positive rate and low accuracy suggest that the **current predictors may not adequately capture the factors influencing order accuracy**. Future efforts should focus on refining the model through better feature selection, potentially using more advanced modeling techniques, and addressing data quality and balance issues. 

## Comparison of Linear Regression and Logistic Regression
- **Model Fit and Accuracy:** Linear Regression is effective for continuous variable prediction, whereas Logistic Regression is suited for binary outcomes but may struggle with accuracy depending on feature relevance and dataset characteristics.

- **Error Metrics:** Linear Regression typically reports errors such as MAE and RMSE to gauge prediction accuracy; Logistic Regression instead utilizes a confusion matrix to illustrate classification accuracy and errors.

- **Predictive Power:** Linear Regression is valuable for modeling and predicting quantitative data; Logistic Regression is essential for binary classification, though its effectiveness hinges on the strength of the relationship between features and the outcome.

- **General Limitations:** Linear Regression requires a linear relationship between variables and is sensitive to outliers; Logistic Regression needs well-defined predictors and is sensitive to imbalanced data, which can skew results significantly.

## References

- [Car Price Prediction Dataset](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)
- [Customer Satisfaction Dataset](https://www.kaggle.com/datasets/ahmedaliraja/customer-satisfaction-10k/data)
- [Machine Learning Module](https://online.fliphtml5.com/grdgl/qwmz/#p=1)

## Members
- [Alojado, Stephen Gabriel S.](https://github.com/t1pen)
- [Sunga, Jairus C.](https://github.com/jairussunga)
