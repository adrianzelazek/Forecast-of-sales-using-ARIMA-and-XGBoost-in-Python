# Forecast of sales using ARIMA and XGBoost in Python
### Author: Adrian Żelazek

In this project was used dataset concering data about sales in stores of some company. The dataset was ceated by merging two datasets. Input dataset contains 1 017 209 observations as well as 18 variables included target variable (Sales). Target varaible is Sales - continuous variable. Before modelling 2 dataset were saved: data_ARIMA and data_XGB. These datasets are the same and was saved because by doing so it is easier to run for example only one model and by doing so it is unnecessarily to firstly run our data engineering and other data transformation, moreover by doing so we can eavoid ventual changes of the dataset during modeling.
The main target of this project was to build, evaluate and compare results of ARIMA and XGBoost models to make forecast of sales for selected stores. 

It is essential to remember that to make good forecast it is important to have enought observations for each Store, in this project was made forecast (using ARIMA and XGBoost) in loop for 10 sample stores (to make comparision fo models) and for 1 store in dedicated build function. Of course it is possible to run function (to make forecast for one Store in one time) or loop (to make forecast for list of Stores in one time) on the whole dataset, but in can consume a lot of time and each Store can be represented by enought observations to make good forcast for this Store.
It is important to emphasize that in some Stores could be not enought observations to make good prediction, and in this case could be overfitting and if some store does not have enought observation, foracast of sales could be even impossible. In this situation is recommended to increase the number of observations for stores that have few observations.

This project was developed for the purpose of practicing machine learning technology and data mining in Python.
Source of dataset: UCI Machine Learning Repository.

In EDA was performed a lot of modifications to prepare dataset to modelling: calculation fo enumerative variables, calculation of days untill and afted holiday, checking data types, duplicates, missings, outliers by boxplots and Isolation Forest, checking distribution of target variable and also logarithm target variable, time series analysis by plots, visualization of seasonality as well as trend per StoreType, dummy coding of categorical varaibles and analysis of CORR by Pearson and Spearman coefficient.

Data selection was performed only based on Spearman CORR, because its does not need normal distribution of variables.
ARIMA
Firstly was created ARIMA model where at the begining were generated ACF and PACF plots. Then was performed Augmented Dickey-Fuller Test to check stationary of series. After that tunning of hiper parameters was made by using loop, where the best (the lowest) AIC and BIC has configuration of order parameters in ARIMA model = 0, 1, 1 and finally this configuraion was used in ARIMA model, because auto_arima consume to much time and finally was not performed (only code was written as comment).

For ARIMA was performed function (to make forecast for one Store in one time) and loop (to make forecast for list of Stores in one time). Evaluaton of model was performed based on: MAE, MSE, RMSE, MAPE. Forecast was save as DF and plots with results on train adn test dataset.

XGBoost
In terms of XGBoost variables: next_holida_date and last_holida_date was removed beause there already exists variables present number of days till and after holiday.

For XGBoost also was performed function (to make forecast for one Store in one time) and loop (to make forecast for list of Stores in one time). Evaluaton of model was performed based on: MAE, MSE, RMSE, MAPE. Forecast was save as DF and plots with results on train adn test dataset.

Generally, results on train and test dataset are similar on ARIMA and XGBoost, so probably there is no overfitting. But is some Stores could be not enought observations to make good prediction, and it this case bould be overfitting and if some store does not have enought observation, foracast of sales could be even impossible.

Comparision of models also was made based on: MAE, MSE, RMSE, MAPE.

##### It is preferred to run/view the project via Jupyter Notebook (.ipynb) than via a browser (HTML).

### Programming language and platform
* Python - version : 3.7.4
* Jupyter Notebook

### Libraries
* Pandas version is 1.1.4
* Scikit-learn is 0.24.1
* Statsmodels is 0.12.1
* Numpy version is 1.20.1
* Matplotlib version is 3.3.3
* Seaborn version is 0.11.0
* XGBoost version is 1.3.3

### Algorithms
* Isolation Forest
* Dummy coding
* log
* Pearson CORR
* Spearman CORR
* Augmented Dickey-Fuller Test
* auto_arima

### Models built
* ARIMA
* XGBoost

### Methods of model evaluation
* MAE
* MSE
* RMS
* MAPE
* comparision of models by DF with statistics and ROC with AUC for each model

