import argparse
import numpy as np
import pandas as pd
import ast
from pycaret.classification import *
from our_utils import preprocess

### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()


# Reading input TSV
data = preprocess(pd.read_csv(args.tsv_path, sep="\t"))
data_no_rev = data.copy().reset_index()
data_no_rev.drop(['revenue'], axis='columns', inplace=True)


model = load_model("extrees")

predictions = predict_model(model, data= data_no_rev)

prediction_df = predictions[['id', 'Label']]
prediction_df.to_csv("prediction.csv", index=False, header=False)



# ### Example - Calculating RMSLE
# res = rmsle(data['revenue'], prediction_df['Label'])
# print("RMSLE is: {:.6f}".format(res))
