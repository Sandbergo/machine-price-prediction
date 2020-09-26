
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from pandas_profiling import ProfileReport


def plot_performance(plot_name, loss_mae, loss_mse):
    steps = np.arange(50, 500, 50)
    plt.style.use('ggplot')
    plt.title(plot_name)
    plt.plot(steps, loss_mae, linewidth=3, label="MAE")
    plt.plot(steps, loss_mse, linewidth=3, label="MSE")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Number of estimators")
    plt.savefig('test.PNG')


def train_random_forest():
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=101)

    plot_name="XGBoosting Regressor"
    loss_mae, loss_mse = [], []
    print(plot_name)
    for est in range(50,500,50):
        print("Number of estimators: %d" % est)
        mae, mse = def_metrics(model.def_xgboost(estimators=est))
        print("MAE: ", mae)
        print("MSE: ", mse)
        loss_mae.append(mae)
        loss_mse.append(mse)
    plot_performance(plot_name, loss_mae, loss_mse)

    return


def load_data():
    data_df = pd.read_csv('data/TrainAndValid.csv', parse_dates=["saledate"])
    df_tmp["saleYear"] = df_tmp.saledate.dt.year
    df_tmp["saleMonth"] = df_tmp.saledate.dt.month
    df_tmp["saleDay"] = df_tmp.saledate.dt.day
    df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
    df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear
    df_tmp.drop("saledate", axis=1, inplace=True)
    return data_df


def analyze_data(data_df, profile=False):
    print(data_df.describe())
    if profile:
        profile = ProfileReport(
            df, title='Pandas Profiling Report', explorative=True
            )
        profile.to_file("figures/report.html")
    plt = df.SalePrice.plot.hist()
    plt.figure.savefig('hist.png')
    data_df.dtypes.value_counts()
    display_all(data_df.dtypes)
    #from fastai.structured import train_cats

    #train_cats(df_raw)

def transfrom_data(raw_df):
    for label, content in df_tmp.items():
        if pd.api.types.is_string_dtype(content):
            df_tmp[label] = content.astype("category").cat.as_ordered()

if __name__ == "__main__":
    df = load_data()
    analyze_data(df)
