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
os.chdir('../data/Sanyo')
warnings.filterwarnings("ignore")

# """
# 1. Extract data according to year
# """
# rawData = pd.read_csv("Sanyo.csv",
#                       usecols=[0, 3, 6, 7, 8, 9],skiprows=[0],
#                       names=["Timestamp", "Power (kW)",
#                              "Temperature (°C)", "Relative Humidity (%)",
#                              "Global Horizontal Radiation (W/m²)",
#                              "Diffuse Horizontal Radiation (W/m²)"])
# rawData.drop(rawData.index[-1])
#
# """
# 2. Add missing Timestamp
# """
# # Correct 'Timestamp' format to datetime
# rawData['Timestamp'] = pd.to_datetime(rawData['Timestamp'])
#
# # Extract 6 years’ data, from 2011 to 2016, (high wind speed missing rate in 2017 and 2018)
# rawData = rawData[(rawData["Timestamp"].dt.year>=2011) & (rawData["Timestamp"].dt.year<=2017)]
#
# # Add missing timestamp
# def addMissingTimestamp(rawData):
#     print("Add missing timestamp")
#     # First 2 timestamps are correct
#     timeDelta5min = rawData.iloc[1, 0] - rawData.iloc[0, 0]
#     for i in trange(rawData.shape[0]-1 ):
#         timeDelta = rawData.iloc[i+1,0]-rawData.iloc[i,0]
#         while timeDelta != timeDelta5min:
#             nDelta = (timeDelta - timeDelta5min)/timeDelta5min
#             rawData = rawData.append({'Timestamp': rawData.iloc[i,0] + timeDelta5min*nDelta}, ignore_index=True)
#             timeDelta = timeDelta - timeDelta5min
#         # if i % 100000 == 0:
#             # print('Add Missing Timestamp Done: %3.1f%%'%(100*i/rawData.shape[0]))
#     return rawData
#
# rawData = addMissingTimestamp(rawData)
# rawData = rawData.sort_values(by='Timestamp')
#
# """
# 3. Extract data according to hourly mean power
# """
# # Extract 12 hours’ data, from 7 o’clock to 18 o’clock
# for i in range(24):
#     print('Mean power (kW) from %2d to %2d o\'çlock : %2.2f '%(i, i+1,
#             np.nanmean(rawData[rawData["Timestamp"].dt.hour==i]['Power (kW)'].values)))
#
# rawData = rawData[(rawData["Timestamp"].dt.hour>=7) & (rawData["Timestamp"].dt.hour<=16)]
#
# # Check data shapes
# # assert (rawData[rawData['Timestamp'].dt.year == 2011].shape[0]==365*12*12)
# # assert (rawData[rawData['Timestamp'].dt.year == 2012].shape[0]==366*12*12)
# # assert (rawData[rawData['Timestamp'].dt.year == 2013].shape[0]==365*12*12)
# # assert (rawData[rawData['Timestamp'].dt.year == 2014].shape[0]==365*12*12)
# # assert (rawData[rawData['Timestamp'].dt.year == 2015].shape[0]==365*12*12)
# # assert (rawData[rawData['Timestamp'].dt.year == 2016].shape[0]==366*12*12)
#
# """
# 4. Find wrong values
# """
# # Caclulate missing rate
# # print('Attrabutes                             Missing Rate (%)')
# # print( 100* (rawData.iloc[:,1:8].isnull().sum() ) / rawData.shape[0] )
# # for i in range(1,3):
# #     print('%s negative values rate: %1.5f%%'%(rawData.columns.values.tolist()[i],100*np.sum((rawData.iloc[:,i]<0) ) / rawData.shape[0]))
# # for i in range(5,7):
# #     print('%s negative values rate: %1.5f%%'%(rawData.columns.values.tolist()[i],100*np.sum((rawData.iloc[:,i]<0) ) / rawData.shape[0]))
# # print('Relative Humidity (%%) wrong values (<0 or >100) rate: %1.5f%%'%(100*np.sum((rawData.iloc[:,4]<0) | (rawData.iloc[:,4]>100)) / rawData.shape[0]) )
#
# # Set wrong values as None
# rawData.iloc[:, 1][(rawData.iloc[:, 1] < 0)] = 0
# rawData.iloc[:, 2][ (rawData.iloc[:,2]<0) ] = None
# for i in range(4,6):
#     rawData.iloc[:, i][ (rawData.iloc[:,i]<0) ] = None
# rawData.iloc[:,3][(rawData.iloc[:,3]<0)] = 0
# rawData.iloc[:,3][(rawData.iloc[:,3]>100)] = 100
#
# # Print out missing rate
# print('Attrabutes                             Missing Rate (%)')
# print( 100* (rawData.iloc[:,1:6].isnull().sum() ) / rawData.shape[0] )
#
# """
# 5. Save data
# """
# rawData.to_csv('ExtractedData.csv',index=False)
#
#
# """
# Impute data by KNN, consider each day as a sample
# """
#
# """
# 1. Load data
# """
# extractedData = pd.read_csv("ExtractedData.csv")
# extractedData['Timestamp'] = pd.to_datetime(extractedData['Timestamp'])
# extractedData = extractedData.set_index(['Timestamp'])
#
# """
# 2. Standardize data
# """
# # Reshape, consider each day as a sample
# extractedData.replace('None', np.nan, inplace=True)
# Data = extractedData.iloc[:,:]
# scaler = StandardScaler()
# scaler.fit(Data)
# Data = scaler.transform(Data)
#
# """
# 3. Impute data
# """
# print('Attrabutes                             Missing Rate (%)')
# print( 100* (extractedData.iloc[:,:].isnull().sum() ) / extractedData.shape[0] )
#
# miceImputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance'),
#                                max_iter=10,tol=1e-3,
#                                n_nearest_features=None,
#                                initial_strategy='mean',imputation_order='ascending',
#                                random_state=0,
#                                verbose=2)
#
# Data = miceImputer.fit_transform(Data)
#
# Data = scaler.inverse_transform(Data)
# extractedData.iloc[:,:] = pd.DataFrame(Data).values
# # Print out missing rate, should be 0
# print('Attrabutes                             Missing Rate (%)')
# print( 100* (extractedData.iloc[:,:].isnull().sum() ) / extractedData.shape[0] )
#
# """
# 4. Save data
# """
# # Inverse-standardize data
# extractedData.to_csv('ImputedData.csv',index=True)
#
#
# """
# 1. Load data
# """
# imputedData = pd.read_csv("ImputedData.csv")
# imputedData['Timestamp'] = pd.to_datetime(imputedData['Timestamp'])
# imputedData = imputedData.set_index(['Timestamp'])
#
# """
# 2. Aggregate data
# """
# aggregatedData = imputedData.resample('30Min').mean()
# # Extract 11 hours’ data, from 7 o’clock to 18 o’clock
# aggregatedData['Timestamp'] = pd.to_datetime(aggregatedData.index)
# aggregatedData = aggregatedData[(aggregatedData["Timestamp"].dt.hour>=7) & (aggregatedData["Timestamp"].dt.hour<=16)]
# aggregatedData = aggregatedData[[ 'Timestamp', 'Power (kW)', 'Temperature (°C)',
#                                   'Relative Humidity (%)', 'Global Horizontal Radiation (W/m²)',
#                                   'Diffuse Horizontal Radiation (W/m²)']]
#
# """
# 3. Reshape data
# """
# dayPVData = np.mean(np.reshape(aggregatedData.iloc[:,1].values,(-1,10*2),order='C'),axis=0)
#
# # fig, ax = plt.subplots()
# # ax.plot(np.arange(7, 17, 0.5),dayPVData)
# # ax.set_xlabel('Time (O\'clock)')
# # ax.set_ylabel('Power (kW)')
# # ax.set_title('Daily average PV power from 7:00am to 4:30pm between 2015 and 2016')
# # plt.show()
# aggregatedData.to_csv('AggregatedData.csv',index=False)


"""
1. Load data
"""
aggregatedData = pd.read_csv("AggregatedData.csv")
aggregatedData['Timestamp'] = pd.to_datetime(aggregatedData['Timestamp'])
aggregatedData = aggregatedData.set_index(['Timestamp'])
print(aggregatedData.shape)
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
st_scaler.fit(data[:,:-3].copy())
data_standardized=data.copy()
data_standardized[:,:-3] = st_scaler.transform(data[:,:-3])
data_scale = st_scaler.scale_
data_mean = st_scaler.mean_
assert (np.allclose(data_standardized[:,:-3]*data_scale+data_mean, data[:,:-3]) == True)
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
# print(data_standardized[:3,:])
print(data.shape)
print(data_standardized.shape)
print(WF.shape)

"""
5. Save data
"""
with h5py.File('Data.h5','w') as H:
    H.create_dataset('X',data=X)
    H.create_dataset('Y',data=Y)
    H.create_dataset('data',data=np.hstack((data,WF*data_scale[-4:]+data_mean[-4:])))
    H.create_dataset('data_scale',data=data_scale)
    H.create_dataset('data_mean',data=data_mean)
    H.create_dataset('noise',data=noise)
