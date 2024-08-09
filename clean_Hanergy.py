import numpy as np
import pandas as pd
from tqdm import trange
import warnings
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import h5py
from scipy import stats
# from fancyimpute import KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import os
os.chdir('../data/Hanergy')
warnings.filterwarnings("ignore")

"""
1. Load data
"""
imputedData = pd.read_csv("ImputedData.csv")
imputedData['Timestamp'] = pd.to_datetime(imputedData['Timestamp'])
imputedData = imputedData.set_index(['Timestamp'])

"""
2. Aggregate data
"""
print(imputedData.values.shape)

aggregatedData = imputedData.resample('30Min').mean()
# Extract 11 hours’ data, from 7 o’clock to 18 o’clock
aggregatedData['Timestamp'] = pd.to_datetime(aggregatedData.index)
print(aggregatedData.values.shape)
aggregatedData = aggregatedData[(aggregatedData["Timestamp"].dt.hour>=7) & (aggregatedData["Timestamp"].dt.hour<=16)]
aggregatedData = aggregatedData[[ 'Timestamp', 'Power (kW)', 'Temperature (°C)',
                                  'Relative Humidity (%)', 'Global Horizontal Radiation (W/m²)',
                                  'Diffuse Horizontal Radiation (W/m²)']]

"""
3. Reshape data
"""
dayPVData = np.mean(np.reshape(aggregatedData.iloc[:,1].values,(-1,10*2),order='C'),axis=0)

# fig, ax = plt.subplots()
# ax.plot(np.arange(7, 17, 0.5),dayPVData)
# ax.set_xlabel('Time (O\'clock)')
# ax.set_ylabel('Power (kW)')
# ax.set_title('Daily average PV power from 7:00am to 4:30pm between 2015 and 2016')
# plt.show()
aggregatedData.to_csv('AggregatedData.csv',index=False)


"""
1. Load data
"""
aggregatedData = pd.read_csv("AggregatedData.csv")
aggregatedData['Timestamp'] = pd.to_datetime(aggregatedData['Timestamp'])
aggregatedData = aggregatedData.set_index(['Timestamp'])
"""
2. Reshape data
"""
numDailyFeatures = 20
numDays = int(aggregatedData.iloc[:,0].values.shape[0]/numDailyFeatures)

def gen_covariates(times):
    covariates = np.zeros((times.shape[0], 3))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.minute
        covariates[i, 1] = input_time.hour
        covariates[i, 2] = input_time.month
    return covariates

covariates = gen_covariates(aggregatedData.index)
data = np.hstack((aggregatedData,covariates))

"""
3. Standardize data
"""
# Standardize PV data
st_scaler = StandardScaler()
st_scaler.fit(data)
data_standardized = st_scaler.transform(data)
data_scale = st_scaler.scale_
data_mean = st_scaler.mean_
assert (np.allclose(data_standardized*data_scale+data_mean, data) == True)

"""
4. Add noise to WF data
"""
# W2: temperature, rainfall and solar irradiance
np.random.seed(0)

noise = 0.2*(np.random.random((data_standardized.shape[0],data_standardized.shape[1]-4)) - 0.5)
WF = data_standardized[:,1:-3] + noise
# WF = np.vstack((WF,np.zeros(WF.shape[1]) ))
X = np.hstack((data_standardized,WF))[0:-20]
Y = data_standardized[20:,0]


"""
5. Save data
"""
with h5py.File('Data.h5','w') as H:
    H.create_dataset('X',data=X)
    H.create_dataset('Y',data=Y)
    H.create_dataset('data',data=np.hstack((data,WF*data_scale[1:5]+data_mean[1:5])))
    H.create_dataset('data_scale',data=data_scale)
    H.create_dataset('data_mean',data=data_mean)
    H.create_dataset('noise',data=noise)
