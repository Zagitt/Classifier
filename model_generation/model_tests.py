import pandas as pd
from ride.unhappened_ride.utils import load_model_from_weights,df_to_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

batch_size = 1   # A small batch sized is used for demonstration purposes
cols = ['id', 'completion', 'is_cc_ride', 'towards_distance', 'get_to_dropoff', 'get_to_pickup',
'pickup_lapse', 'driver_called', 'passenger_called',
'has_spike', 'linear_distance', 'service', 'is_approached','output']
df = pd.read_csv('model_generation_input_test.csv',header=None,names=cols)

input = df.drop(['service','id'],axis=1)
batch_size = 1   # A small batch sized is used for demonstration purposes
X = np.asarray(input).astype(np.float32)
sc = StandardScaler()

X = sc.fit_transform(X)

model = load_model_from_weights('/Users/giuseppemarotta/Documents/rpa-project/ride/unhappened_ride/model_generation/model_weights/',shape=12)

res = pd.DataFrame(model.predict(X))

res.to_csv('results.csv')