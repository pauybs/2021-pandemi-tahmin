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

covidDataFrame = pd.read_json("items3.json")  

covidDataFrame = covidDataFrame.drop("sourceUrl",axis=1)
covidDataFrame = covidDataFrame.drop("readMe",axis=1)
covidDataFrame = covidDataFrame.drop("lastUpdatedAtSource",axis=1)
covidDataFrame= covidDataFrame.drop("dailyTested",axis=1)
covidDataFrame= covidDataFrame.drop("dailyInfected",axis=1)
covidDataFrame= covidDataFrame.drop("dailyDeceased",axis=1)
covidDataFrame = covidDataFrame.drop("dailyRecovered",axis=1)
covidDataFrame = covidDataFrame.drop("critical",axis=1)
covidDataFrame = covidDataFrame.drop("ICU",axis=1)
covidDataFrame= covidDataFrame.drop("lastUpdateresultpify",axis=1)

covidDataFrame = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-12"))]

covidDataFrame = covidDataFrame.fillna(covidDataFrame.mean())

#ARIMA prediciton model
"""
covidDataFrame["lastUpdatedAtApify"] = pd.to_datetime(covidDataFrame["lastUpdatedAtApify"])
covidDataFrame = covidDataFrame.set_index("lastUpdatedAtApify")

from pmdarima import auto_arima 
aa = auto_arima(covidDataFrame['infected'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()
#print(aa)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
train_data = covidDataFrame[:len(covidDataFrame)-12]
test_data = covidDataFrame[len(covidDataFrame)-12:]
arima_model = SARIMAX(train_data['infected'], order = (2,1,1), seasonal_order = (4,0,3,12))
arima_result = arima_model.fit()
ar = arima_result.summary()
print(ar)
arima_pred = arima_result.predict(start = len(train_data), end = len(covidDataFrame)-1, typ="levels").rename("ARIMA Predictions")
#test_data['infected'].plot(figsize = (100,90),legend=True)
#arima_pred.plot(legend = True)
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

arima_rmse_error = rmse(test_data['infected'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = covidDataFrame['infected'].mean()

print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')

covidDataFrame.index.freq = 'MS'

#ax = covidDataFrame['infected'].plot(figsize = (16,5), title = "infected")
ar.set(xlabel='Dates', ylabel='Infected');

"""



"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(covidDataFrame["infected"].values)
covidDataFrame["infected"].values = imputer.transform(covidDataFrame["infected"].values)
"""



#Tıme-series analysis forecasting

"""
covidDataFrame.lastUpdatedAtApify = pd.to_datetime(covidDataFrame.lastUpdatedAtApify)
covidDataFrame['lastUpdatedAtApify'] = covidDataFrame['lastUpdatedAtApify'].dt.tz_localize(None)
#print(covidDataFrame.tail())
trace = go.Scatter(x=list(covidDataFrame.lastUpdatedAtApify),
                   y=list(covidDataFrame.infected), line=dict(color='red'))
dat = [trace]
layout = dict(
    title='Zaman Serisi Analizi',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=dat, layout=layout)
#plotly.offline.plot(fig)
from fbprophet import Prophet
import fbprophet 
df = covidDataFrame.rename(columns={'lastUpdatedAtApify': 'ds', 'infected': 'y'})

fbp = fbprophet.Prophet()
fbp.fit(df)
df_forecast = fbp.make_future_dataframe(periods=24,freq='M')
df_forecast = fbp.predict(df_forecast)
fbp.plot_components(df_forecast)
fbp.plot(df_forecast, xlabel = 'Date', ylabel = 'Infected')
plt.title('Zaman Serisi Analizi-2')
plt.show()



def is_summer_season(ds):
    date = pd.to_datetime(ds)
    return (date.month < 9 and date.month > 5)



df['on_season'] = df['ds'].apply(is_summer_season)
df['off_season'] = ~df['ds'].apply(is_summer_season)




m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')
"""
"""
df['on_season'] = df['ds'].apply(is_summer_season)
df['off_season'] = ~df['ds'].apply(is_summer_season)
"""
"""
forecast = m.fit(df).predict(df)
fig = m.plot_components(forecast)

forecast.to_excel(r"forecast-1.xlsx",index="false")
df.to_excel(r"df-1.xlsx",index="false")

expected = df["y"].values
predicted = forecast["yhat"].values

forecast_errors = [expected[i]-predicted[i] for i in range(len(expected))]
bias = sum(forecast_errors) * 1.0/len(expected)
print('Bias: %f' % bias)

#LMST training and test data prediction model
"""

"""
from sklearn.preprocessing import MinMaxScaler


covidDataFrame['lastUpdatedAtApify'] = pd.to_datetime(covidDataFrame.lastUpdatedAtApify, format='%Y-%m-%d')
covidDataFrame.index = covidDataFrame.lastUpdatedAtApify
values = covidDataFrame['infected'].values.reshape(-1,1)
#values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)

TRAIN_SIZE = 0.60
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Gün Sayıları (training set, test set): " + str((len(train), len(test))))

def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)
# Yeni verisetinin şekline bakalım.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()
    # Modelin tek layerlı şekilde kurulacak.
    model.add(LSTM(100, 
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam")
   #30 epoch yani 30 kere verisetine bakılacak.
    model.fit(train_X, 
              train_Y, 
              epochs = 80, 
              batch_size = 1, 
              verbose = 1)
    
    return(model)
# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict_and_score(model, X, Y):
    # Şimdi tahminleri 0-1 ile scale edilmiş halinden geri çeviriyoruz.
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    # Rmse değerlerini ölçüyoruz.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)
rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)
print("Training MAPE: %.2f" % mean_absolute_percentage_error(train_Y,train_predict))
print("Training data score: %.2f RMSE" % rmse_train)
print("Test MAPE: %.2f" % mean_absolute_percentage_error(test_Y,test_predict))
print("Test data score: %.2f RMSE" % rmse_test)

train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict
# Şimdi ise testleri tahminletiyoruz.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict
# Plot'u oluşturalım.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Days")
plt.ylabel("Exchange Rates")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()

forecast_errors = [test_Y[i]-test_predict[i] for i in range(len(test_Y))]
bias = sum(forecast_errors) * 1.0/len(test_Y)
print('Test data Bias: %f' % bias)

forecast_errors = [train_Y[i]-train_predict[i] for i in range(len(train_Y))]
bias = sum(forecast_errors) * 1.0/len(train_Y)
print('Training data Bias: %f' % bias)

"""

# Artificial neural network -relu,sigmoid- model up to line 365

"""covidDataFrame['lastUpdatedAtApify']= pd.to_datetime(covidDataFrame['lastUpdatedAtApify'])
covidDataFrame = covidDataFrame.set_index('lastUpdatedAtApify')
"""
#dt = datetime.datetime.strptime("%Y-%m-%d")
"""summerData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-09")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-05"))]
winterData = covidDataFrame[(covidDataFrame["lastUpdatedAtApify"]<("2020-12")) & (covidDataFrame["lastUpdatedAtApify"]>("2020-08"))]

summerData = summerData.drop("lastUpdatedAtApify",axis=1)
winterData = winterData.drop("lastUpdatedAtApify",axis=1)


summerData = summerData.drop("lastUpdatedAtSource",axis=1)
winterData = winterData.drop("lastUpdatedAtSource",axis=1)

summerData= summerData.drop("dailyTested",axis=1)
summerData= summerData.drop("dailyInfected",axis=1)

summerData= summerData.drop("dailyDeceased",axis=1)
summerData= summerData.drop("dailyRecovered",axis=1)
summerData= summerData.drop("critical",axis=1)
summerData= summerData.drop("ICU",axis=1)

winterData = winterData.drop("critical",axis=1)

winterData = winterData.drop("ICU",axis=1)
summerData= summerData.drop("lastUpdateresultpify",axis=1)
winterData = winterData.drop("lastUpdateresultpify",axis=1)
winterData= winterData.drop("dailyTested",axis=1)
winterData= winterData.drop("dailyInfected",axis=1)

winterData= winterData.drop("dailyDeceased",axis=1)
winterData= winterData.drop("dailyRecovered",axis=1)
"""
#y = winterData["infected"].values

"""
y = (winterData["infected"]-winterData["infected"].min())/(winterData["infected"].max()-winterData["infected"].min())
print(y)
X = winterData.drop("infected",axis=1)
print(X)
"""

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=10)
"""
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train) #Eğitim setine normalizasyon uygulamak
X_test = mms.transform(X_test) #Test setine normalizasyon 

"""
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
"""

"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
classifier.add(Dense(2,kernel_initializer=initializer,activation="relu",input_dim=3)) 
classifier.add(Dense(2,kernel_initializer=initializer,activation="relu"))
classifier.add(Dense(1,kernel_initializer=initializer,activation="sigmoid"))
classifier.compile(optimizer="adam",loss="MSE", metrics=[tf.keras.metrics.Accuracy()])
classifier.fit(X_train,y_train,epochs=50)
y_pred=classifier.predict(X_test)
"""

"""
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
"""

"""
sc = classifier.score(X_test, y_pred)

print(sc)
"""

"""
from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))

predictions = [round(value) for value in y_pred]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_pred, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
"""

"""
kayıpVeri = pd.DataFrame(classifier.history.history)
kayıpVeri.plot()

plt.show()
"""   

# Artificial neural network -relu- model up to line 456

"""
summerData = summerData.drop("lastUpdatedAtApify",axis=1)
winterData= winterData.drop("lastUpdatedAtApify",axis=1)

summerData = summerData.drop("lastUpdatedAtSource",axis=1)
winterData = winterData.drop("lastUpdatedAtSource",axis=1)
summerData= summerData.drop("dailyTested",axis=1)
summerData= summerData.drop("dailyInfected",axis=1)

summerData= summerData.drop("dailyDeceased",axis=1)
summerData= summerData.drop("dailyRecovered",axis=1)
summerData= summerData.drop("critical",axis=1)
summerData= summerData.drop("ICU",axis=1)

winterData = winterData.drop("critical",axis=1)

winterData = winterData.drop("ICU",axis=1)
summerData= summerData.drop("lastUpdateresultpify",axis=1)
winterData = winterData.drop("lastUpdateresultpify",axis=1)
winterData= winterData.drop("dailyTested",axis=1)
winterData= winterData.drop("dailyInfected",axis=1)

winterData= winterData.drop("dailyDeceased",axis=1)
winterData= winterData.drop("dailyRecovered",axis=1)


print(summerData.describe())
print(summerData.isnull().sum(),np.any(np.isnan(summerData)))
y = (winterData["infected"]-winterData["infected"].min())/(winterData["infected"].max()-winterData["infected"].min())
x = summerData.drop("infected",axis=1).values
#y = summerData["infected"].values
#x = summerData.drop("infected",axis=1).values

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)
#print(len(X_train),len(X_test))
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_train.shape)

model = Sequential()
model.add(Dense(3,activation="relu"))
model.add(Dense(3,activation="relu"))
model.add(Dense(3,activation="relu"))
model.add(Dense(3,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")

model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=80)

kayıpVeri = pd.DataFrame(model.history.history)
kayıpVeri.plot()

plt.show()
"""


"""
cvscores = []
scores = model.evaluate(X_test, y_test, verbose=0)
print((model.metrics_names, scores*100))
cvscores.append(scores * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

accuracy_score(y_test, y_pred)
"""

"""
from sklearn.model_selection import cross_val_score
success = cross_val_score(estimator = model,X=X_train,y=y_train,cv=4)
print(success.mean())
"""

"""

tahminDizisi = model.predict(X_test)
mean_absolute_error=mean_absolute_error(y_test,tahminDizisi)
print(mean_absolute_error)
plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")

r2_Deep = r2_score(y_test,tahminDizisi)
print("r2_deep_learn",r2_Deep)

plt.show()

"""

# Linear Regression model 
"""
lin = linear_model.LinearRegression()
X = (summerData["infected"]-summerData["infected"].min())/(summerData["infected"].max()-summerData["infected"].min())
y = (winterData["infected"]-winterData["infected"].min())/(winterData["infected"].max()-winterData["infected"].min())
#print(X,y)
plt.scatter(X.values,y.values)
plt.xlabel('x')
plt.ylabel('Y')
plt.title('X(Yaz verisi) y(kış verisi) arasındaki ilişki')
plt.show()
#plt.show()"""
"""lineer_regresyon = LinearRegression()
lineer_regresyon.fit(X.values.reshape(-1,1),y.values.reshape(-1,1))
print("Elde edilen regresyon modeli: Y={}+{}X".format(lineer_regresyon.intercept_,lineer_regresyon.coef_[0]))
y_predicted = lineer_regresyon.predict(X.values.reshape(-1,1))
r2 = r2_score(y,y_predicted)
print("Model ",round(r2,4)," oranında verilere uyum sağladı.")
print("Ortalama Mutlak Hata: {} \nOrtalama Karesel Hata: {}".format(
 mean_absolute_error(y, y_predicted), mean_squared_error(y, y_predicted)))
#print(f1_score())
random_x = np.array([0, 0.5, 0.99])
plt.scatter(X.values, y.values,color="blue")
plt.plot(random_x,
         lineer_regresyon.intercept_[0] +
         lineer_regresyon.coef_[0][0] * random_x,
         color='red',
         label='regresyon grafiği',
         linewidth = "2")
plt.xlabel('x')
plt.ylabel('Y')
plt.title('X y regresyon analizi')  
plt.show()
"""
"""plt.scatter(summerData["lastUpdatedAtApify"],summerData["infected"])
plt.xlabel('Tarih')
plt.ylabel('Enfekte Sayısı')
plt.title('Yaz Ayları Enfekte Sayısı Grafiği')
plt.show()

plt.scatter(winterData["lastUpdatedAtApify"],winterData["infected"],color="blue")
plt.xlabel('Tarih')
plt.ylabel('Enfekte Sayısı')
plt.title('Kış Ayları Enfekte Sayısı Grafiği')
plt.show()"""
"""lin.fit(Xsample, ysample)
t0, t1 = lin.intercept_[0], lin.coef_[0][0]
print(t0,t1)

locSummer = summerData["infected"]
predicted = lin.predict( locSummer)[0]
print(predicted)
"""
        