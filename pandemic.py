import plotly
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
import keras

#import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sbn
import plotly.plotly as py
import warnings
import plotly.graph_objs as go 
import seaborn as sns
from datetime import datetime
import numpy
import math
import tensorflow as tf
import seaborn as sns

df = pd.read_excel("gun-il-ort.xlsx")
covidDataFrame = pd.read_json('items3.json')  

covidDataFrame = covidDataFrame.drop("sourceUrl",axis=1)
covidDataFrame = covidDataFrame.drop("readMe",axis=1)
covidDataFrame = covidDataFrame.drop("lastUpdatedAtSource",axis=1)
#covidDataFrame= covidDataFrame.drop("lastUpdateresultpify",axis=1)


"""

covidDataFrame = covidDataFrame.drop("lastUpdatedAtSource",axis=1)
covidDataFrame = covidDataFrame.drop("historyData",axis=1)
covidDataFrame = covidDataFrame.drop("activeCases",axis=1)
covidDataFrame = covidDataFrame.drop("critical",axis=1)
covidDataFrame= covidDataFrame.drop("lastUpdateresultpify",axis=1)
"""
"""
covidDataFrame= covidDataFrame.drop("hospitalDeceased",axis=1)
covidDataFrame= covidDataFrame.drop("hospitalized",axis=1)
covidDataFrame= covidDataFrame.drop("newlyHospitalized",axis=1)
covidDataFrame = covidDataFrame.drop("intensiveCare",axis=1)
covidDataFrame = covidDataFrame.drop("newlyIntensiveCare",axis=1)
covidDataFrame = covidDataFrame.drop("recoverd",axis=1)
"""


covidDataFrame= covidDataFrame.drop("dailyTested",axis=1)
covidDataFrame= covidDataFrame.drop("dailyInfected",axis=1)
covidDataFrame= covidDataFrame.drop("dailyDeceased",axis=1)
covidDataFrame = covidDataFrame.drop("dailyRecovered",axis=1)
covidDataFrame = covidDataFrame.drop("critical",axis=1)
covidDataFrame = covidDataFrame.drop("ICU",axis=1)

xx = df.iloc[:, : -1].values
yy = df.iloc[:,2]
labelemcoder_X = LabelEncoder()
xx[:, 0] = labelemcoder_X.fit_transform(xx[:,0])
df["AQI"] = xx[:, 0]

#print(df.corr())
#sns.pairplot(df,hue = "AQI")
#sns.heatmap(df.corr(), annot=True, lw=1
"""
#france has some missing values until 2020-07-01
if covidDataFrame["infected"] == Null:
    covidDataFrame = covidDataFrame.drop("infected",axis=1)
"""
covidDataFrame = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-12-31"))]
"""  
covidDataFrame = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]>("2020-07-01"))&
                                 ((covidDataFrame["lastUpdatedAtApify"]<("2021-01-01")))]
"""
    
#iran has missing values until 2020-10

#for iran dataset
"""
if covidDataFrame["tested"].values == "N/A":
    covidDataFrame["tested"] = 0
  """  
covidDataFrame = covidDataFrame.fillna(covidDataFrame.mean())

"""
#Linear Regression
summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-06"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2021-01")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-09"))]

x_summer = df[(df["Tarih"]<("2020-09")) & (df["Tarih"]>("2020-06"))]
x_winter = df[(df["Tarih"]<("2021-01")) & (df["Tarih"]>("2020-09"))]

#winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-11")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#X PM10, y infected

#kış
#X = x_winter["Ortalama"].values
#y = winterData["infected"].values
#yaz


X = x_summer["Ortalama"].values
y = summerData["infected"].values
"""
"""
#yaz
X = summerData["infected"].values
y = x_summer["AQI"].values 
"""
#kış
#X = winterData["infected"].values
#y = x_winter["AQI"].values 
"""
length = len(X)
X = X.reshape((length,1))
length1 = len(y)
y = y.reshape((length1,1))
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)

plt.scatter(X,y,color="red")
plt.title("Kış Veriler")
plt.xlabel("X")
plt.ylabel("y")
plt.show()


X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
lineer_regresyon = LinearRegression()
lineer_regresyon.fit(X_train,y_train)
print("Elde edilen regresyon modeli: Y={}+{}X".format(lineer_regresyon.intercept_,lineer_regresyon.coef_[0]))
y_pred = lineer_regresyon.predict(X_test)
y_pred_train = lineer_regresyon.predict(X_train)

plt.scatter(X_train,y_train,color="red", label='Gerçek Veri')
plt.plot(X_train, lineer_regresyon.predict(X_train),
         color = "blue",
         label='Tahmin Verisi',
         linewidth = "2")
plt.xlabel('COVID-19 Vaka Sayısı ')
plt.ylabel('AQI')
plt.title('Yaz Mevsimi COVID-19 Vaka Sayısı Ve AQI Verileri (Eğitim)')
plt.show()


#random_x = np.array([0, 0.5, 0.99])
plt.scatter(X_test, y_test, color="red", label='Gerçek Veri')
plt.plot(X_train, lineer_regresyon.predict(X_train),
         color = "blue",
         label='Tahmin Verisi',
         linewidth = "2")
plt.xlabel('COVID-19 Vaka Sayısı ')
plt.ylabel('AQI')
plt.title('Yaz mevsimi COVID-19 Vaka Sayısı Ve AQI Verileri (Test)')
plt.show()

from sklearn.metrics import median_absolute_error
from math import sqrt

print("Test\n")
print("R-Kare: ",r2_score(y_test, y_pred))
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))
print("MedAE: ",median_absolute_error(y_test, y_pred))

print("Eğitim\n")
print("R-Kare: ",r2_score(y_train,  y_pred_train))
print("MAE: ",mean_absolute_error(y_train, y_pred_train))
print("MSE: ",mean_squared_error(y_train, y_pred_train))
print("RMSE: ",sqrt(mean_squared_error(y_train, y_pred_train)))
print("MedAE: ",median_absolute_error(y_train, y_pred_train))

"""
"""

plt.scatter(summerData["lastUpdatedAtApify"],summerData["infected"])
plt.xlabel('Tarih')
plt.ylabel('Enfekte Sayısı')
plt.title('Yaz Ayları Enfekte Sayısı Grafiği')
plt.show()

plt.scatter(winterData["lastUpdatedAtApify"],winterData["infected"],color="blue")
plt.xlabel('Tarih')
plt.ylabel('Enfekte Sayısı')
plt.title('Kış Ayları Enfekte Sayısı Grafiği')
plt.show()
"""

# Meteorological Artifical Neural Network
"""
summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-06"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2021-01")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-09"))]

x_summer = df[(df["Tarih"]<("2020-09")) & (df["Tarih"]>("2020-06"))]
x_winter = df[(df["Tarih"]<("2021-01")) & (df["Tarih"]>("2020-09"))]

#winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-11")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#X PM10, y infected
"""
"""
#kış
X = x_winter["Ortalama"].values
y = winterData["infected"].values
"""
#yaz
#X = x_summer["Ortalama"].values
#y = summerData["infected"].values
"""

#yaz
X = summerData["infected"].values
y = x_summer["Ortalama"].values 

#kış
#X = winterData["infected"].values
#y = x_winter["Ortalama"].values 

length = len(X)
X = X.reshape((length,1))
length1 = len(y)
y = y.reshape((length1,1))
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)

"""
"""
plt.scatter(X,y,color="red")
plt.title("Yaz Veriler")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
"""
"""
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

initializer = tf.keras.initializers.HeUniform()

classifier.add(Dense(12, kernel_initializer=initializer, activation = "relu", input_dim = 1))

classifier.add(Dense(12, kernel_initializer=initializer, activation = "relu"))

classifier.add(Dense(1, kernel_initializer=initializer, activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss="MSE")

history = classifier.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size= 12, epochs= 150)

y_pred_ANN = classifier.predict(X_test)

lossData = pd.DataFrame(history.history)
lossData.plot()
plt.show()

from sklearn.metrics import median_absolute_error

print("R-Kare: ",r2_score(y_test, y_pred_ANN))
print("MAE: ",mean_absolute_error(y_test, y_pred_ANN))
print("MSE: ",mean_squared_error(y_test, y_pred_ANN))
print("MedAE: ",median_absolute_error(y_test, y_pred_ANN))

for i in classifier.layers:
    first_layer = classifier.layers[0].get_weights()
    second_layer = classifier.layers[1].get_weights()
    output_layer = classifier.layers[2].get_weights()   

scaler_inverse = scaler.inverse_transform(y_pred_ANN.reshape(-1,1))

"""

# Meteorological Recurrence Neural Network (LSTM)

"""
summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-06"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2021-01")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-09"))]

x_summer = df[(df["Tarih"]<("2020-09")) & (df["Tarih"]>("2020-06"))]
x_winter = df[(df["Tarih"]<("2021-01")) & (df["Tarih"]>("2020-09"))]

#winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-11")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#X PM10, y infected
"""
#kış
#X = x_winter["Ortalama"].values
#y = winterData["infected"].values
"""
#yaz
X = x_summer["Ortalama"].values
y = summerData["infected"].values
"""
"""

#yaz
#X = summerData["infected"].values
#y = x_summer["Ortalama"].values 

#kış
X = winterData["infected"].values
y = x_winter["Ortalama"].values 

length = len(X)
X = X.reshape((length,1))
length1 = len(y)
y = y.reshape((length1,1))
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)
"""
"""

plt.scatter(X,y,color="red")
plt.title("Yaz Veriler")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
"""
"""
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)


	
# reshape input to be [samples, time steps, features]
X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

from keras.models import Sequential
from keras.layers import Dense

from keras.layers import LSTM
from keras.layers import Dropout



classifier = Sequential()


classifier.add(LSTM(6,
               input_shape=(1,1),     
               return_sequences=True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(4,return_sequences=True))

classifier.add(Dropout(0.2))



classifier.add(Dense(1))

#model.compile(loss='mse', optimizer='rmsprop')

classifier.compile(optimizer = "adam", loss="MSE")
#yaz
history = classifier.fit(X_train,
          y_train,
          batch_size=4,
         validation_data=(X_test,y_test),
          verbose=1,
          epochs=75,
          )
#kış
"""
"""
history = classifier.fit(X_train,
          y_train,
          batch_size=4,
         validation_data=(X_test,y_test),
          verbose=1,
          epochs=75,
          )
"""
"""
y_pred_LSTM = classifier.predict(X_test)

y_pred_LSTM_train = classifier.predict(X_train)

lossData = pd.DataFrame(history.history)
lossData.plot()
plt.show()

nsamples, nx, ny = y_pred_LSTM.shape
test_dataset = y_pred_LSTM.reshape((nsamples,nx*ny))

"""
"""
nsamples1, nx1, ny1 = y_pred_LSTM_train.shape
train_dataset = y_pred_LSTM_train.reshape((nsamples1,nx1*ny1))

trainPredictPlot = numpy.empty_like(df.values)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[1:len(y_pred_LSTM_train)+1, ] = train_dataset
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df.values)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(y_pred_LSTM_train)+(1*2)+1:len(df.values)-1, :] = test_dataset
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df.values))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
"""

"""
from sklearn.metrics import median_absolute_error
from math import sqrt

print("R-Kare: ",r2_score(y_test, test_dataset))
print("MAE: ",mean_absolute_error(y_test, test_dataset))
print("MSE: ",mean_squared_error(y_test, test_dataset))
print("RMSE: ",sqrt(mean_squared_error(y_test, test_dataset)))
print("MedAE: ",median_absolute_error(y_test, test_dataset))

for i in classifier.layers:
    first_layer = classifier.layers[0].get_weights()
    second_layer = classifier.layers[2].get_weights()
    output_layer = classifier.layers[4].get_weights() 
    

scaler_inverse = scaler.inverse_transform(y_pred_LSTM.reshape(-1,1))
"""

#SVM
"""
summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-06"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2021-01")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-09"))]

x_summer = df[(df["Tarih"]<("2020-09")) & (df["Tarih"]>("2020-06"))]
x_winter = df[(df["Tarih"]<("2021-01")) & (df["Tarih"]>("2020-09"))]

#winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-11")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#X PM10, y infected
"""
#kış
"""
X = x_winter["Ortalama"].values
y = winterData["infected"].values
"""
#yaz
#X = x_summer["Ortalama"].values
#y = summerData["infected"].values
"""

#yaz
X = summerData["infected"].values
y = x_summer["AQI"].values 

#kış
#X = winterData["infected"].values
#y = x_winter["Ortalama"].values 


length = len(X)
X = X.reshape((length,1))
length1 = len(y)
y = y.reshape((length1,1))
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)
"""
"""
X_train = scaler.fit_transform(X_train.reshape(-1,1))
X_test = scaler.fit_transform(X_test.reshape(-1,1))
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.fit_transform(y_test.reshape(-1,1))
"""
"""

from sklearn.svm import SVC

classifier = SVC(kernel = "rbf", random_state = 0, C=1.0, degree=3, gamma="auto")
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)   

from matplotlib import colors
from matplotlib.colors import ListedColormap

colors1 = ["#DBDBDB","#DCD5CC","#DCCEBE","#DDC8AF","#DEC2A0","#DEBB91", "#DFB583", "#DFAE74", "#E0A865", "#E1A256", "#E19B48", "#E29539"]
colors2 = colors.ListedColormap(colors1)

def evaluationCriteria(report, title = None, cmap = colors2):
    lines =  report.split("\n")
    classLabels = []
    lists = []
    
    for i in lines[2:len(lines)-3]:
        s = i.split()
        classLabels.appends(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        lists.appends(value)

    fig, ax = plt.subplot(1)
    
    for column in range(len(lists)+1):
        txt = lists[lines][column]
        ax.text(column, lines, lists[lines][column],va="center", ha = "center")
        
    fig = plt.imshow(lists, interpolation = "nearest", cmap = cmap)
    plt.title("Sınıflandırma İçin Değerlendirme Ölçütleri")
    plt.colorbar()
    X_ticks = np.arange(len(classLabels)+1)
    y_ticks = np.arange(len(classLabels))
    plt.xticks(X_ticks, ["Kesinlik", "Duyarlılık", "F-Skoru"],rotation = 45)
    plt.yticks(y_ticks, classLabels)
    plt.ylabel("Sınıflar")
    plt.xlabel("Ölçütler")
    plt.show()
    
report = classification_report(y_test, y_pred)
evaluationCriteria(report)

scaler_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
"""

#SVR

"""
summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-06"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2021-01")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-09"))]

x_summer = df[(df["Tarih"]<("2020-09")) & (df["Tarih"]>("2020-06"))]
x_winter = df[(df["Tarih"]<("2021-01")) & (df["Tarih"]>("2020-09"))]

#winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-11")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#X PM10, y infected


#kış
X = x_winter["Ortalama"].values
y = winterData["infected"].values

#yaz
#X = x_summer["Ortalama"].values
#y = summerData["infected"].values
"""
"""
#yaz
X = x_summer["Ortalama"].values 
y = summerData["infected"].values 

#kış
#X = x_winter["Ortalama"].values 
#y = winterData["infected"].values 
"""
"""
length = len(X)
X = X.reshape((length,1))
length1 = len(y)
y = y.reshape((length1,1))
scaler.fit(X)
X = scaler.transform(X)
scaler.fit(y)
y = scaler.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#from sklearn.svm import SVR

model = SVR(kernel = "rbf", C=1.0, degree=2, gamma="auto", epsilon = 0.1)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
#Modelin grafiğinin çizilmesi

#Eğitim

X_grid = np.arange(min(X_train),max(X_train),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_grid, model.predict((X_grid)),color="blue")
plt.title("Destek Vektör Regresyonu Yaz Eğitim Verileri")
plt.xlabel("PM10")
plt.ylabel("COVID-19 Vaka Sayısı")
plt.show()


#Test

X_grid = np.arange(min(X_test),max(X_train),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_grid, model.predict((X_grid)),color="blue")
plt.title("Destek Vektör Regresyonu Yaz Test Verileri")
plt.xlabel("PM10")
plt.ylabel("COVID-19 Vaka Sayısı")
plt.show()


from sklearn.metrics import median_absolute_error
from math import sqrt
#print("Elde edilen regresyon modeli: Y={}+{}X".format(SVR.intercept_,SVR.coef_[0]))
print("Test\n")
print("R-Kare: ",r2_score(y_test,  y_pred))
print("MAE: ",mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))
print("MedAE: ",median_absolute_error(y_test, y_pred))
print("Eğitim\n")
print("R-Kare: ",r2_score(y_train,  y_pred_train))
print("MAE: ",mean_absolute_error(y_train, y_pred_train))
print("MSE: ",mean_squared_error(y_train, y_pred_train))
print("RMSE: ",sqrt(mean_squared_error(y_train, y_pred_train)))
print("MedAE: ",median_absolute_error(y_train, y_pred_train))

scaler_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
"""
