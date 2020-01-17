import pandas as pd
import numpy as np
import glob
from sklearn.externals import joblib


def load_all_csv_files():
    Y_df = []
    X_df = []
    for file in glob.glob("Res/CompaniesTrainingwithoutbase/*csv"):
        csv = pd.read_csv(file, index_col=0)
        Y_df.append(csv["Change Stock Value"])
        X_df.append(csv.drop(["Change Stock Value"], axis=1))
    Y_train = np.array(pd.concat(Y_df, axis=0))
    X_train = np.array(pd.concat(X_df, axis=0))
    return X_train, Y_train


def save_scalers(scalerX, scalerY):
    scalerX_filename = "Results/scalerX.save"
    scalerY_filename = "Results/scalerY.save"
    joblib.dump(scalerX, scalerX_filename)
    joblib.dump(scalerY, scalerY_filename)


def load_scalers():
    scalerX_filename = "Results/scalerX.save"
    scalerY_filename = "Results/scalerY.save"
    scalerX = joblib.load(scalerX_filename)
    scalerY = joblib.load(scalerY_filename)
    return scalerX, scalerY
