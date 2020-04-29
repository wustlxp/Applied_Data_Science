import numpy as np
import pandas as pd
import copy
from matplotlib import pylab as plt
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


def drop_duplicates(data):
    """
    function: drop duplicates
    param: DataFrame
    return: new DataFrame without duplicates
    """
    return data.drop_duplicates(keep='first')


def month_sales(data):
    """
    function: transfer daily sales to monthly sales
    param: DataFrame
    return: DataFrame with 'item_cnt_month' instead of 'item_cnt_day'
    """
    col = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    data = data[col].groupby(["item_id", "shop_id", "date_block_num"]).agg(
        {'item_price': 'mean', 'item_cnt_day': 'sum'}).reset_index()
    data.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
    return data


def drop_outliers(data):
    """
    function: caculate zscore of the average of monthly sales, and drop samples whose average is over 3 or below -3.
    Hint: Normally, we don't drop outliers of target variable.
    However, in this case, the problem statement true target values are clipped into [0,20].
    And some the target values have extreme high value, which affects our accuracy.
    param: DataFrame
    return: DataFrame without outliers
    """
    y = data['item_cnt_month']
    zscore = (y - np.mean(y)) / np.std(y)
    drop_index = zscore[(zscore > 3) | (zscore < -3)].index
    data.drop(drop_index, axis=0, inplace=True)
    return data


def time_series(data):
    """
    function: create historical sales records columns for the previous months
    param: DataFrame
    return: DataFrame with time series columns
    """
    table = data.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_month',
                             aggfunc='sum').fillna(0.0).reset_index()
    table['shop_id'] = table.shop_id
    table['item_id'] = table.item_id
    df_price = df[['shop_id', 'item_id', 'item_price']].groupby(["item_id", "shop_id"]).mean().reset_index()
    data = pd.merge(table, df_price, on=['shop_id', 'item_id'], how='inner')
    return data


def item_category(data):
    """
    function: extract new feature item_category from 'item_id'
    param: DataFrame
    return: DataFrame with new column 'new_item_category'
    """
    item_category_id = pd.read_csv(r'C:\kaggle\items.csv')
    item_category = pd.read_csv(r'C:\kaggle\item_categories.csv')
    df_category = pd.merge(item_category_id, item_category, on=['item_category_id'])
    df_category['item_category_id'].astype(np.int32)
    df_category['new_item_category'] = 'other'
    df_category['new_item_category'].loc[
        df_category['item_category_id'].between(1, 8, inclusive=True)] = 'Digital Appliances'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(10, 18, inclusive=True)] = 'Consoles'
    df_category['new_item_category'].loc[
        df_category['item_category_id'].between(18, 25, inclusive=True)] = 'Consoles Games'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(28, 31, inclusive=True)] = 'CD games'
    df_category['new_item_category'].loc[
        df_category['item_category_id'].between(26, 27, inclusive=True)] = 'Phone games'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(32, 36, inclusive=True)] = 'Card'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(37, 42, inclusive=True)] = 'Movie'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(43, 54, inclusive=True)] = 'Books'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(55, 60, inclusive=True)] = 'Music'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(61, 72, inclusive=True)] = 'Gifts'
    df_category['new_item_category'].loc[df_category['item_category_id'].between(73, 79, inclusive=True)] = 'Soft'
    data = pd.merge(data, df_category[['item_id', 'new_item_category']], on=['item_id'], how='left')

    return data


def dummy(data):
    """
    function: one-hot encoding for categorical columns
    param: DataFrame
    return: DataFrame with one-hot encoding
    """
    df_cate = pd.get_dummies(data['new_item_category'], drop_first=True)
    data = pd.concat([data, df_cate], axis=1)
    data.drop(['new_item_category'], axis=1, inplace=True)
    return data


def standarization(data):
    """
    function: standarization for columns 'item_price'
    param: DataFrame
    return: Standarized DataFrame
    """
    scaler = StandardScaler()
    col = ['item_price']
    data[col] = scaler.fit_transform(data[col])
    return data


def drop_y(data):
    """
    function: keep predictors (X) and drop target variable (y)
    param: DataFrame
    return: predictors only
    """
    return data.drop(33, axis=1)


def transform_test_set(data):
    """
    function: move all monthly sales forward one month.
    Hint: Since the sales volume of month 34 is predicted,
          other sales volumes should be one month in advance to accommodate the model.
    param: DataFrame
    return: transformed DataFrame
    """
    col = list(range(0, 33))
    test = copy.deepcopy(data)
    test[col] = df[np.add(col, 1)].values
    test[33] = 0
    return test


def calculate_real_target(df):
    """
    function: Our real target variable is monthly sales, not daily sales.
              You have to run this function before you can get the real X and the real Y.
    Hint: This process must be outside the pipeline.
          Because during this process, the number of rows changes significantly.
          If placed in a pipeline, the dimensions of X and y will not match.
    param: DataFrame
    return: DataFrame with the real target variable 'item_sales_month'
    """
    df = drop_duplicates(df)
    df = month_sales(df)
    df = drop_outliers(df)
    df = time_series(df)
    return df