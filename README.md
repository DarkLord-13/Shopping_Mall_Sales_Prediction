# Big Mart Sales Prediction

This project aims to predict the sales of items based on their various attributes using machine learning techniques. The notebook is implemented in Python and utilizes libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Dataset

The dataset used in this project is the Big Mart Sales dataset, which includes information on items and their sales in different outlets.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Usage

1. Import the necessary libraries:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor
    from sklearn import metrics
    ```

2. Load the dataset:
    ```python
    dataset = pd.read_csv('/content/Train.csv')
    ```

3. Display the first few rows of the dataset:
    ```python
    dataset.head()
    ```

4. Handle missing values:
    ```python
    dataset['Item_Weight'].fillna(dataset['Item_Weight'].mean(), inplace=True)
    mode_of_outlet_size = dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
    missing_values = dataset['Outlet_Size'].isnull()
    dataset.loc[missing_values, 'Outlet_Size'] = dataset.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])
    ```

5. Perform data analysis:
    ```python
    dataset.describe()
    ```

6. Visualize numerical features:
    ```python
    plt.figure(figsize=(4,4))
    sns.distplot(dataset['Item_Weight'])
    plt.show()

    plt.figure(figsize=(4,4))
    sns.distplot(dataset['Item_Visibility'])
    plt.show()

    plt.figure(figsize=(4,4))
    sns.distplot(dataset['Item_MRP'])
    plt.show()

    plt.figure(figsize=(8,4))
    sns.countplot(x=dataset['Outlet_Establishment_Year'], data=dataset)
    plt.show()
    ```

## Google Colab

You can run this notebook on Google Colab using the following link:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DarkLord-13/Machine-Learning-01/blob/main/BigMartSalesPrediction.ipynb)

## Author

Nishant Kumar

Feel free to fork the project and raise an issue if you have any questions or suggestions.
