
import pandas as pd


def load_data():
    
    data_df = pd.read_csv('data/TrainAndValid.csv')
    print(data_df.describe())



if __name__ == "__main__":
    load_data()