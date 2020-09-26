"""
XGBoost regressor for construction machine price prediction
"""
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from pandas_profiling import ProfileReport


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

plt.style.use('ggplot')


def train_linear_regressor(x_train, x_test, y_train, y_test):
    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The mean squared error
    print(f'Mean squared error: {mean_squared_error(y_test, y_pred)}')
    print(f'Mean absolute error: {mean_absolute_error(y_test, y_pred)}')

    ax = sns.residplot(x=y_test, y=y_pred)
    ax.set(xlabel='residual', ylabel='salgspris', title='Residualplot')

    plt.savefig("figures/linear_residuals.png")


def load_data():
    data_df = pd.read_csv(
        'data/TrainAndValid.csv',
        parse_dates=["saledate"],
        low_memory=False
        )
    return data_df


def analyze_data(data_df, profile=False):
    if profile:
        print(data_df.describe())
        profile = ProfileReport(
            data_df, title='Pandas Profiling Report', explorative=True
            )
        profile.to_file("figures/report.html")
    data_df.dtypes.value_counts()


def transform_data(raw_df):
    raw_df["saleYear"] = raw_df.saledate.dt.year
    raw_df["saleMonth"] = raw_df.saledate.dt.month
    raw_df["saleDay"] = raw_df.saledate.dt.day
    raw_df["saleDayOfWeek"] = raw_df.saledate.dt.dayofweek
    raw_df["saleDayOfYear"] = raw_df.saledate.dt.dayofyear
    raw_df.drop("saledate", axis=1, inplace=True)

    for label, content in raw_df.items():
        if pd.api.types.is_string_dtype(content):
            raw_df[label] = content.astype("category").cat.as_ordered()

        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            raw_df[label+"_is_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1
            raw_df[label] = pd.Categorical(content).codes + 1

        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing
                raw_df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                raw_df[label] = content.fillna(content.median())

    return raw_df


def train_xgb_regressor(x_train, y_train):
    """ Creates and trains a XGB-model on the given data
    Args:
        train_X:       training set input
        train_y:       training set labels
        validation_X = validation set input
        validation_y = validation set labels
    Returns:
        xgb_model    = XGB-model trained on given data
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=101
        )

    xgb_model = xgb.XGBRegressor(
        base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=1, gamma=0, importance_type='gain',
        learning_rate=0.1, max_delta_step=0, max_depth=9,
        min_child_weight=1, missing=None, n_estimators=1000, n_jobs=-1,
        nthread=None, objective='reg:squarederror', random_state=101,
        reg_alpha=2, reg_lambda=0.2, scale_pos_weight=1,
        seed=101, silent=False, subsample=1
        )

    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=8,
        verbose=True
        )
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Test')
    ax.legend()
    plt.ylabel('Mean Average Error')
    plt.title('Mean Average Error for XGB')
    plt.savefig('figures/xgb_training')
    # xgb_model.load_model('../models/model_finn.hdf5')
    xgb_model.save_model('models/xgb.hdf5')
    return xgb_model


def feature_importances(xgb_model, train_X):
    """ prints the importances of features 
    Args:
        xgb_model:     XGB-model
        train_X:       training set
    Returns:
    """
    print('Feature importances in descending order:')

    input_features = train_X.columns.values
    feat_imp = xgb_model.feature_importances_
    np.split(feat_imp, len(input_features))
    feat_imp_dict = {}
    for i in range(0, len(input_features)):
        feat_imp_dict[feat_imp[i]] = input_features[i]

    sorted_feats = sorted(feat_imp_dict.items(), key=operator.itemgetter(0))
    for i in range(len(sorted_feats) - 1, 0, -1):
        print(sorted_feats[i])
    print()
    return


if __name__ == "__main__":
    raw_df = load_data()
    analyze_data(raw_df)
    transformed_df = transform_data(raw_df)
    x_train, x_test, y_train, y_test = train_test_split(
        transformed_df.drop('SalePrice', axis=1),
        transformed_df.SalePrice,
        test_size=0.2,
        random_state=101
        )
    train_linear_regressor(x_train, x_test, y_train, y_test)

    train_xgb_regressor(transformed_df.drop('SalePrice', axis=1),
        transformed_df.SalePrice)
