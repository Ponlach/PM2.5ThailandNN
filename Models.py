#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Data preparation
import pandas as pd
import os
import numpy as np 

#Load pollution dataset
start = 2019
end = 2023
df = pd.read_excel(f'pm/2.5/PM2.5({start}).xlsx')
for i in range(start+1,end+1):
    tmp = pd.read_excel(f'pm/2.5/PM2.5({i}).xlsx')
    df = pd.concat([df, tmp],join='inner', ignore_index=True)


# In[5]:


#Drop bad stations data
df = df.dropna(thresh=round(df.shape[1]*0.1))
df = df.dropna(axis=1,thresh=round(df.shape[0]*0.1))


# In[6]:


#Interpolate NA in pm dataset
df_raw = df
df = df_raw.drop('Date', axis=1)
mean_date = df.mean(axis=1)
mean_station = list(df.mean(axis=0)/np.nanmean(df))
df_mean = pd.DataFrame(np.outer(mean_date, mean_station), columns = list(df.columns)).astype(float).round(1)
df = df.fillna(df_mean)
df = df.interpolate()
dfscaled = (df-df.mean())/df.std()


# In[7]:


#Add time features to pm dataset
import datetime
import matplotlib as mpl

missing = []
for nc in os.listdir('inputs/aod'):
    missing.append(nc[23:30])
missing.sort()

df_raw['id_date'] = missing

dfday = df_raw['id_date'].str.strip().str[-3:].astype(int)
df['cos_day'] = np.cos(dfday * (2 * np.pi / (365.2425)))
dfscaled['cos_day'] = np.cos(dfday * (2 * np.pi / (365.2425)))


# In[8]:


#Load satellite dataset
#!pip install netCDF4
import packaging 
import netCDF4
from matplotlib import pyplot as plt

def input_arr_thai(file_name):
    ds = netCDF4.Dataset('inputs/aod/'+file_name)
    columns_raw = []
    for key in ds.variables.keys():
        if ('Mean' in key or 'Standard_Deviation' in key): #select only mean features
            columns_raw.append(key)

    list_df2 = []
    tmp = []
    col_name = []
    for col in columns_raw:
        tmp = ds[col]
        if tmp.shape == (180 , 360):
            tmp2 = []
            for i in ds[col][94:113]:
                tmp2.append(i[274:288].tolist())
            list_df2.append(tmp2)
            col_name.append(col)
            
    #arr_df2 = np.array(list_df2)
    return list_df2, col_name, file_name[23:30]


# In[9]:


df2_details = []
for nc in os.listdir('inputs/aod'):
    df2_details.append(list(input_arr_thai(nc)))
    print(nc)
df2 = [i[0] for i in df2_details]


# In[10]:


# # #Pickle!
# import pickle

# with open('df2_detailslist8.pickle', 'wb') as handle:
#     pickle.dump(df2_details, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('df2_detailslist8.pickle', 'rb') as handle:
#     b = pickle.load(handle)
# # print(df2_details == b)
# df2_details = b
# df2 = [i[0] for i in df2_details]


# In[11]:


#Features example plots
nc = netCDF4.Dataset('inputs/aod/'+'AERDB_D3_VIIRS_NOAA20.A2019141.002.2023083214506.nc')
plt.imshow(nc['Angstrom_Exponent_Land_Ocean_Mean'])
plt.show()


# In[12]:


import plotly.express as px
fig = px.histogram(df, x="02T", nbins=20)
fig.update_layout(title_text='PM2.5 levels')
fig.show()


# In[13]:


where = pd.read_excel("locations.xlsx", header=None)
where[['lat','lon']] = where[2].str.split(', ', expand=True)

wheres = where.drop(2,axis=1)
wheres['lat'] = wheres['lat'].astype(float)
wheres['lon'] = wheres['lon'].astype(float)


# In[15]:


#!pip install folium
import folium
from folium.features import DivIcon

map_center = [wheres['lat'].mean(), wheres['lon'].mean()]
map_ = folium.Map(location=map_center, zoom_start=5, control_scale=True)

# Add markers for each location in the DataFrame
for i, row in wheres.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row[0]
    ).add_to(map_)


# Display the map
map_.save("map.html")
map_


# In[16]:


date_time = pd.to_datetime(df_raw.pop('Date'), format='%Y.%m.%d %H:%M:%S')
plot_cols = ['02T', '05T', '10T']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)


# In[18]:


df


# In[22]:


#Scale the satellite dataset
import statistics

varall = []
fea = 14 #33
for v in range(0,fea):
    varsum = []
    varsumn = []
    var = []
    for i in df2[0:1]:
        var += i[v]
    varsumn = sum(var, [])
    varsum = [i for i in varsumn if i is not None]
    train_mean = np.mean(varsum)
    train_std = np.std(varsum)
    train_max = np.max(varsum)
    train_min = np.min(varsum)
    if train_max == 0:
        train_max = 1
    varall+=[[train_mean,train_std]]

listy = df.values.tolist()
listx = []

n=0
for i in df2[1:]:#
    listx += [[-99 if v == None else v for v in sum(sum(i, []),[])]+listy[n]]
    n+=1
    if n >= len(listy):
        break
listy2 = df.drop('cos_day', axis=1).values.tolist()[1:]#

listx2 = listx
for i in range(len(listx2)):
    att = 0
    for j in range(0,fea):
        for k in range(0,266):
            if listx2[i][att] == -99:
                listx2[i][att] = 0
            else:     
                listx2[i][att] = (listx[i][att]-varall[j][0])/varall[j][1]
            att+=1

#Transform to arrays
X=np.array(listx2)
y=np.array(listy2)
print(f"X shape: {X.shape}, Y shape: {y.shape}")


# In[28]:


#Evaluate model matric
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def percen10(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)<=10)/(y_true.shape[0]*y_true.shape[1])

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


# In[25]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,SimpleRNN
from sklearn.model_selection import train_test_split

patience=20
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                mode='min')


# In[27]:


#Baseline model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)


n = X_train.shape[1]  # number of input features
m = y_train.shape[1]   # number of output features

model = Sequential()
model.add(Dense(m,input_dim=X_train.shape[1],activation=None))                                   

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()

history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2, callbacks=[early_stopping])


# In[29]:


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
p10 = percen10(y_test, y_pred)
print(mae, mse, r2)


# In[188]:


#ANN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
n = X_train.shape[1]  
m = y_train.shape[1]  

model = Sequential()
model.add(Dense(256, input_dim=n, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(m))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=300, batch_size=8, validation_split=0.2,callbacks=[early_stopping])


# In[189]:


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
p10 = percen10(y_test, y_pred)
print(mae, mse, r2)


# In[149]:


# for i in range(4):
i = 0
j = 2 #station
dsta = 0
dend = 60
d = np.arange(1, dend-dsta+1)
d2 = np.transpose(y_pred[dsta:dend,j:j+1])
d3 = np.transpose(y_test[dsta:dend,j:j+1])
plt.plot(d, d2[i], marker='o', label=f'pred')
plt.plot(d, d3[i], marker='o', label=f'test')

# Adding labels and title
plt.xlabel('Day')
plt.ylabel('PM2.5')
plt.title('prediction at a station')
plt.legend()

# Display the plot
plt.show()


# In[183]:


wheres = where.drop(2,axis=1)
wheres['lat'] = wheres['lat'].astype(float)
wheres['lon'] = wheres['lon'].astype(float)
wheres['pred'] = y_pred[0,:].astype(float).round(1)
wheres['test'] = y_test[0,:]

map_center = [wheres['lat'].mean(), wheres['lon'].mean()]
map_ = folium.Map(location=map_center, zoom_start=5, control_scale=True)

# Add markers for each location in the DataFrame
for i, row in wheres.iterrows():
    yp = row['pred']
    yt = row['test']
    folium.Marker(
        location=[row['lat'], row['lon']],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(7,20),
            html=f'<div style="font-size: 10pt; color : blue"><mark>{yp}</mark></div><div style="font-size: 12pt; color : red">{yt}</div>'
        )
    ).add_to(map_)


# Display the map
map_.save("map.html")
map_


# In[35]:


#Prepare data for lstm by convert series to supervised learning
def window_slice(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    reslice = pd.concat(cols, axis=1)
    reslice.columns = names
    if dropnan:
        reslice.dropna(inplace=True)
    return reslice


# In[36]:


#Set time windows
output_days = 3-1 #target-1
input_days = 7-1 #day-1


arr_X = window_slice(X[:-output_days], input_days,1)
arr_y = window_slice(y[input_days:], 1,output_days)

y2 = arr_y.values
X2 = arr_X.values.reshape((y2.shape[0],input_days+1,X.shape[1]))


# In[38]:


#For spicific date
y2 = arr_y.values[:,y.shape[1]*output_days:]


# In[39]:


print(X[:-output_days].shape,y[input_days:].shape,X.shape,y.shape,arr_X.shape,arr_y.shape)


# In[191]:


#Split train test val for lstm
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, shuffle = False)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle = False)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[193]:


patience=10
epoch= 100
batchs = 16
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                mode='min')
from tensorflow.keras.layers import SimpleRNN

model2 = Sequential()
model2.add(LSTM(256, activation='relu',stateful=False, batch_input_shape=(batchs,X_train.shape[1], X_train.shape[2])))
model.add(Dense(256, activation='relu'))
model2.add(Dense(y_train.shape[1]))

model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model2.summary()



history2 = model2.fit(X_train, y_train, epochs=epoch, batch_size=batchs, validation_data=(X_val, y_val), callbacks=[early_stopping])


# In[194]:


y_pred = model2.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
p10 = percen10(y_test, y_pred)
print(mae, mse, r2)


# In[208]:


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# In[201]:


y_test[0:20,0:5]


# In[210]:


y_pred[:,0:5].round(2)


# In[207]:


y_preda[1:21,:].round(2)


# In[190]:


y_preda = model.predict(X_test)



# In[192]:


model = Sequential()
model.add(LSTM(units, stateful=True, batch_input_shape=(batch_size, time_steps, input_dim)))
model.add(Dense(1))  # Example output layer, say we want to predict the closing price

model.compile(optimizer='adam', loss='mse')

# Generate random target data
target = np.random.random((num_samples, 1))

# Train the model
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    # Train the model for each batch
    for i in range(num_samples // batch_size):
        x_batch = data[i * batch_size:(i + 1) * batch_size]
        y_batch = target[i * batch_size:(i + 1) * batch_size]
        model.train_on_batch(x_batch, y_batch)
    # Reset states after each epoch
    model.reset_states()


