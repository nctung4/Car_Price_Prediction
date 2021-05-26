# Car Price Prediction

This project is a end-to-end data science project, from data collecting, to making it into production.

### Project Overview
* Created a tool that estimates car prices (RMSE:~293K, MAE:~293K HUF) to help people negotiate or estimate the specific price, when they want to either buy or sell a car.
* Used web scraping techniques to get the data from [hasznaltauto.hu](https://www.hasznaltauto.hu/). (hungarian car trading website)
* Applied several data cleaning and feature engineering techniques, such as handling missing data or feature scaling
* Optimized multiple linear regression, support vector regression, random forest regression, and xgboost.
* Build a client facing API using flask, and deployed it using heroku (PAAS): https://nctung-car-price-pred.herokuapp.com/

### Resources used
**Python Version:** 3.8.5 <br>
**Packages:** pandas, numpy, seaborn, matplotlib, sickit-learn, xgboost, selenium, beautifulsoup4, Flask, gunicorn, requests <br>
**Requirements:** ```pip install -r requirements.txt```  
 
### Table of contents:
1. [Project Motivation](#project-motivation)
2. [Technical Aspects](#technical-aspects)
3. [Major Insights](major-insights)
    * 3.1 [Web scraping](#web-scraping)
    * 3.2 [Data preparation](#data-preparation)
    * 3.3 [EDA](#eda)
    * 3.4 [Model building](#model-building)
    * 3.5 [Flask Deployment](#flask-deployment)
4. [Credits](#credits)

### Project Motivation
I am not a car expert, and I always wondered how much my car would worth at the moment, if I suddenly decide to sell it. Then this idea hit me from nowhere. Why not build a model which help me figure out this information? I am a beginner in web scraping, it was quite hard at the beginning, but after a few readings, I managed to get the data I need from the biggest car trading website in my country. In the website I set a filtering for age and price. I searched for cars based on my specific preferences. Their price was maximum 3.5M HUF, and year ranges from 2011 to 2019. Unfortunately, my dataset is very small, due to the fact that my computer does not have the computing capacity, to run the code through a whole night. I scraped the data from the first 50 pages. I was also curious about how should a model like this be deployed. One of the most popular solution for that, is by a flask web application. I decided to create it as well, and host it in a PAAS service.

### Technical Aspects
* I scraped the data using the **selenium** and **beautifulsoup** libraries.
* I cleaned the data using **pandas**.
* I used several visualization techniques, to perform EDA.
* In the model building part I optimized 4 models. **Linear Regression**, **SVR**, **Random Forest Regression**, and **XGBoost**.
* Build a client facing API using **Flask**, and deployed it in **heroku**.

### Major Insights
In this section, I want to show the major insights of this project, from collecting the data, to deploying it.

#### Web scraping
In the website, 1 car information looks like this:

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/web_scrape_page.png)

I collected 350 cars data, looped through 50 pages. I managed to make the looping by changing the URL query parameters. I used the dictionary data structure to collect these data, and after 1 car I appended the dict into a list. 

#### Data preparation

Several data cleaning procedures was applied in this phase, such as parsing the data to the right type, or getting the essential information within 1 cell.

The notebook for this section can be found [here].(https://github.com/nctung4/Car_Price_Prediction/blob/main/1_Data_preparation.ipynb)

#### EDA
In this section the main goal is to summarize the variables main characteristics. One of the most important thing to check at first, is the distribution of the numerical variables, and the relationship with the other variables through a scatter plot.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/pairplot.png)

We can see that most of the variables are either normally distributed, or have a right skewed distribution. 

The next thing I wanted to check, is the correaltion with each other.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/corr_plot.png)

We can see that the weight variables have at least a moderate correlation with the other variables, we might need to exclude this from the modeling part, to avoid multicollinearity.

One of the simplest way to check the categorical variable characteristics, is to apply bar chart.

There are several imbalance variables in the dataset, such as the document, which should be excluded in the modeling part. 

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/docment_bar.png)

I also checked that what kind of manufacturers are available in this specific price range.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/manufacturer.png)

For more information check the [notebook].(https://github.com/nctung4/Car_Price_Prediction/blob/main/2_EDA.ipynb)

#### Model building
In the model building part I tried 4 models. 

Evaluation summary table:

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/eval.png)

The Linear Regression had the best performance, with an approximately 70% adjusted R^2.

In case of linear regression, it is essential after the modeling, to check whether the 5 assumptions are met in our case. The 5 assumptions are:
1. Linear relationship
2. Normality 
3. No or little multicollinearity
4. No auto-correlation
5. Homoscedasticity

**Linear relationship:**

In this case, between the target and the feature variables, there should be a linear relationship. We can check that if the actual and the predicted values follow a diagonal line.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/linearitycheck.png)

In our case this assumption is satisfied.

**Normality of the error terms:**

The error term should follow a normal distribution. We can use statistical tests, such as the Anderson-Darling test, to check for normal distribution. If the p<=0.05 (we reject the null hypothesis), then the distribution is non-normal. In our case this assumption is also satisfied.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/normality.png)

**Multicollinearity:**

With the VIF indicators, we can check whether in the specific variable multicollinearity may be present or not. If the VIF is greater than 10, then multicollinearity might be present. Unfortunately, this assumption is not satisfied in that case. However, it was not surprising, because we have a lot of dummy variables, and in that case, multicollinearity cannot be avoided. 

**No auto-correlation in the error terms:**

Durbin-Watson Test shows if there is a autocorrelation in the data.
Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data.

In our case the Durbin-Watson value is 1.93, so this assumption is satisfied as well.

**Homoscedasticity:**

It basically means that the error terms should have the same variance in each case. So the variance should be constant.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/homoscedasticiy.png)

We can say that in our case this assumption is also satisfied, so we can use the linear regression model, to predict the car price.

#### Flask Deployment

I created a flask web application, where the user have to fill in the specific data, and after pressing the calculate button, the estimated car price will be displayed below the button.

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/frontend_app.png)

![](https://github.com/nctung4/Car_Price_Prediction/blob/main/plots/pred_app.png)

I deployed this application in heroku, it is available here: https://nctung-car-price-pred.herokuapp.com/

### Credits:
* https://medium.com/swlh/web-scrapping-healthcare-professionals-information-1372385d639d
* https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
* https://stackabuse.com/deploying-a-flask-application-to-heroku/
* https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/
* https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
* Also grateful to [Ken Jee](https://www.linkedin.com/in/kenjee/) and [Krish Naik](https://www.linkedin.com/in/naikkrish/?originalSubdomain=in) for showing me how to create an end-to-end project 
