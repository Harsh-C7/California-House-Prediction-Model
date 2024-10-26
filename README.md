# California Housing Price Prediction

This project is focused on predicting housing prices in California using machine learning techniques. The dataset used for this project is the California Housing dataset, which is a popular dataset for regression tasks. The primary goal is to build a predictive model that can accurately estimate the price of a house based on various features.

## Project Overview

The project involves the following key steps:

1. **Data Loading and Exploration**: 
   - The California Housing dataset is loaded using `fetch_california_housing` from [`sklearn.datasets`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html).
   - The dataset is converted into a [Pandas](https://pandas.pydata.org/) DataFrame for easier manipulation and analysis.
   - Basic exploratory data analysis (EDA) is performed, including checking for null values and generating descriptive statistics.

2. **Data Preprocessing**:
   - The correlation between features is visualized using a heatmap to understand relationships within the data.
   - Features are standardized using [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to ensure that they have a mean of 0 and a standard deviation of 1, which is crucial for many machine learning algorithms.

3. **Model Training**:
   - The dataset is split into training and testing sets using [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
   - An [`XGBRegressor`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor) model from the [XGBoost library](https://xgboost.readthedocs.io/en/stable/) is used for training. XGBoost is known for its efficiency and performance in regression tasks.

4. **Model Evaluation**:
   - The model's performance is evaluated using metrics such as R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE) for both training and testing datasets.
   - Scatter plots are generated to visualize the relationship between predicted and actual values for both training and testing datasets.

## Results

The model achieves a high R-squared value on the training data, indicating a good fit. However, the performance on the test data is slightly lower, which is typical and suggests that the model generalizes well but could potentially be improved further.

## Visualization

The project includes visualizations such as correlation heatmaps and scatter plots to provide insights into the data and the model's predictions.

## Conclusion

This project demonstrates the application of machine learning techniques to predict housing prices. It highlights the importance of data preprocessing, model selection, and evaluation in building effective predictive models. The use of XGBoost showcases its capability in handling regression tasks efficiently.

## Future Work

Future improvements could include hyperparameter tuning, feature engineering, and exploring other regression models to enhance prediction accuracy. Additionally, deploying the model as a web application could provide a user-friendly interface for real-time predictions.
        
