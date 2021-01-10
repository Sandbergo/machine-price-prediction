"""
XGBoost regressor for construction machine price prediction
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport

pd.set_option('display.max_rows', 500)
plt.style.use('ggplot')


def load_data(path: str) -> pd.DataFrame:
    """loads data

    Args:
        path (str): path to data

    Returns:
        pd.DataFrame: DataFrame of data
    """
    data_df = pd.read_csv(
        path,
        parse_dates=["saledate"],
        low_memory=False
        )
    return data_df


def analyze_data(data_df: pd.DataFrame, profile_title='') -> None:
    """analyze data and make Pandas Profile

    Args:
        data_df (pd.DataFrame): Data
        profile_title (str, optional): Profile title. Defaults to ''.
    """
    print(data_df.describe())
    data_df.dtypes.value_counts()
    if profile_title != '':
        profile = ProfileReport(
            data_df, title=profile_title, explorative=True)
        profile.to_file(f"figures/{profile_title.split(' ', 1)[0]}_report.html")


def transform_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform data to ML-ready format

    Args:
        raw_df (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Transformed data
    """
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
            raw_df[label+"_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1
            raw_df[label] = pd.Categorical(content).codes + 1

        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing
                raw_df[label+"_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                raw_df[label] = content.fillna(content.median())

    return raw_df


def train_xgb_regressor(x_train: pd.DataFrame, y_train: pd.DataFrame):
    """ Creates and trains a XGB-model on the given data
    Args:
        x_train:       training set input
        y_train:       training set labels
    Returns:
        xgb_model: XGB-model trained on given data
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=101
        )
    xgb_model = xgb.XGBRegressor(
        base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=1, gamma=0, importance_type='gain',
        learning_rate=0.2, max_delta_step=0, max_depth=13,
        min_child_weight=10, missing=None, n_estimators=5, n_jobs=-1,
        nthread=None, objective='reg:squarederror', random_state=101,
        reg_alpha=2, reg_lambda=0.2, scale_pos_weight=1,
        seed=101, subsample=1
        )
    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=8,
        verbose=True
        )
    # xgb_model.load_model('models/xgb.hdf5')
    results = xgb_model.evals_result()
    x_axis = range(0, len(results['validation_0']['mae']))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_axis, results['validation_0']['mae'], label='Train')
    ax.plot(x_axis, results['validation_1']['mae'], label='Test')
    ax.set_ylim(0, 10_000)
    ax.legend()
    plt.ylabel('Mean Average Error')
    plt.title('Mean Average Error for XGB')
    plt.savefig('figures/xgb_training')
    plt.clf()
    xgb_model.save_model('models/xgb.hdf5')
    return xgb_model


def compare_prediction_to_benchmark(
        y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame
        ) -> None:
    """Compares model test-set performance to median benchmark

    Args:
        y_train (pd.DataFrame):
        y_test (pd.DataFrame):
        y_pred (pd.DataFrame):
    """
    test_eval = {}
    test_eval['pred'] = y_pred
    test_eval['benchmark'] = y_train.median()
    test_eval['target'] = y_test.reset_index(drop=True)
    test_eval['diff'] = test_eval['pred'] - test_eval['target']
    test_eval['bench diff'] = test_eval['benchmark'] - test_eval['target']
    test_eval['abs diff'] = abs(test_eval['diff'])
    test_eval['abs bench diff'] = abs(test_eval['bench diff'])
    test_eval['diff %'] = (test_eval['pred'] / test_eval['target'] - 1) * 100
    test_eval['bench diff %'] = abs((test_eval['benchmark'] / test_eval['target'] - 1) * 100)

    mean = int(test_eval['abs diff'].mean())
    bench_mean = int(test_eval['abs bench diff'].mean())
    mean_perc = round(abs(test_eval['diff %']).mean(), 2)
    bench_mean_perc = round(abs(test_eval['bench diff %']).mean(), 2)
    print('Model evaluation compared to median benchmark:')
    print('\n                        | our model | benchmark')
    print(f'| mean abs.  difference | {mean}      |{bench_mean}')
    print(f'| mean abs % difference | {mean_perc} %   |{bench_mean_perc} %\n')

    ax = sns.residplot(x=y_test, y=y_pred)
    ax.set(xlabel='sale price', ylabel='residual', title='Residualplot')

    plt.savefig("figures/residuals.png")
    plt.clf()


def plot_features(columns: list, importances: list, n=20) -> None:
    """Create plot of feature importances

    Args:
        columns (list): columns in data
        importances (list): importances from model
        n (int, optional): number of features. Defaults to 20.
    """
    df = (pd.DataFrame({"features": columns,
                        "feature_importance": importances})
          .sort_values("feature_importance", ascending=False)
          .reset_index(drop=True))

    sns.barplot(x="feature_importance",
                y="features",
                data=df[:n],
                orient="h")
    plt.savefig("figures/features.png")
    plt.clf()


def main():
    print('Loading data...')
    raw_df_trainval = load_data('data/TrainAndValid.csv')
    raw_df_test = load_data('data/Test.csv')
    raw_df = raw_df_trainval.append(raw_df_test)
    # analyze_data(raw_df, profile_title='Raw Data Profile')

    transformed_df = transform_data(raw_df)
    
    transformed_df = transformed_df.dropna(subset=['SalePrice'])
    print(transformed_df.shape)
   
    correlations = transformed_df.apply(lambda x: x.corr(transformed_df['SalePrice']))
    print(f'correlations: \n{correlations}')
    # analyze_data(transformed_df, profile_title='Transformed Data Profile')

    x_train, x_test, y_train, y_test = train_test_split(
        transformed_df.drop('SalePrice', axis=1),
        transformed_df.SalePrice,
        test_size=0.1,
        random_state=101
        )
    print('Training model...')
    xgb_model = train_xgb_regressor(
        x_train=x_train,
        y_train=y_train
        )

    plot_features(x_train.columns, xgb_model.feature_importances_)

    y_pred = xgb_model.predict(x_test)

    compare_prediction_to_benchmark(y_train=y_train, y_test=y_test, y_pred=y_pred)
    print('Finished sucessfully')


def predict_price(ID: str):
    xgb_model = xgb.XGBRegressor() 
    xgb_model.load_model('models/xgb.hdf5')

    raw_df_trainval = load_data('data/TrainAndValid.csv')
    raw_df_test = load_data('data/Test.csv')
    
    raw_df = raw_df_trainval.append(raw_df_test)
    
    # analyze_data(raw_df, profile_title='Raw Data Profile')

    transformed_df = transform_data(raw_df)
    
    x_id = transformed_df.loc[transformed_df['SalesID'] == ID].drop('SalePrice', 1)
   
    predicted_price = xgb_model.predict(x_id)
    
    return float(predicted_price)


if __name__ == "__main__":
    main()
