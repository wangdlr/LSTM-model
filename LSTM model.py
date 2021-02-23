from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import TimeDistributed
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pandas import read_csv
from numpy import concatenate
import pywt
#LSTM model to forecast the next 20 steps
#rolling forecasting method
# load data
series = read_csv('200014pressure--.csv', header=None,index_col=0)
# prepare data for normalization
values = series.values
print(values)
values=values.astype('float32')
# train the normalization
values=values[1:]
scaler = MinMaxScaler(feature_range=(0.1,0.9))
scaler = scaler.fit(values)
# normalize the dataset and print
normalized = scaler.transform(values)
normalized=DataFrame(normalized)
# split into train and test sets
reframed=concat([normalized,normalized.shift(-1)],axis=1)
reframed=concat([reframed,normalized.shift(-2)],axis=1)
print(reframed.head())
x=reframed.values
train_size = int(len(x) * 0.7)
train, test = x[0:train_size,:], x[train_size:len(x)-2,:]
train_x, train_y = train[:,0:8], train[:,8:12]
test_x, test_y = test[:,0:8], test[:,8:12]
train_x= train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x= test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(x.shape,train.shape,test.shape)

# define model
model = Sequential()
model.add(LSTM(32, input_shape=(train_x.shape[1],train_x.shape[2])))
model.add(Dense(4, activation= 'linear'))
# compile model
model.compile(loss='mse', optimizer='adam')
# fit model
history = model.fit(train_x, train_y, epochs=300, validation_data=(test_x, test_y),shuffle=False)
print(model.summary())

# plot train and validation loss
pyplot.figure(1)
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
#pyplot.plot(history.history['acc'])
#pyplot.plot(history.history['val_acc'])
pyplot.title('model train and validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train','validation'], loc='upper right')
pyplot.show()
predictions=model.predict(test_x)
test_x=test_x.reshape((test_x.shape[0],test_x.shape[2]))
test11=predictions
test11=scaler.inverse_transform(test11)
input1=test_x

#prediction for the next 20 steps
n=20
if n>1:
  for i in range(1,n):
    input1=input1.reshape(len(input1),8)
    input1=DataFrame(input1)
    predictions=DataFrame(predictions)
    input1=concat([input1,predictions],axis=1)
    input1=DataFrame(input1)
    input1=input1.values
    input1=input1[:,-8:]
    input1=input1.reshape(len(input1),1,8)
    predictions=model.predict(input1,verbose=0)
    predictions=predictions.reshape(len(predictions),4)
    test12=predictions
    test12=scaler.inverse_transform(test12)
    test11=DataFrame(test11)
    test12=DataFrame(test12)
    test11=concat([test11,test12],axis=1)
predictions=scaler.inverse_transform(predictions)
prediction1=predictions[:,0]
prediction2=predictions[:,1]
test_y=test_y.reshape(len(test_y),4)
test_y=scaler.inverse_transform(test_y)
observation1=test_y[:,0]
observation2=test_y[:,1]

#prediction on typhoon case
series1=read_csv('201509pressure--.csv',header=None,index_col=0)
values1=series1.values
scaler1=MinMaxScaler(feature_range=(0.1,0.9))
scaler1=scaler1.fit(values1)
normalized1=scaler1.transform(values1)
values11=DataFrame(normalized1)
values1=concat([values11,values11.shift(-1)],axis=1)
test=values1.values
test=test[:len(test)-1,:]
test=test.reshape(test.shape[0],1,test.shape[1])
testresult=model.predict(test,verbose=0)
test=test.reshape(test.shape[0],test.shape[2])
test21=testresult
test21=scaler1.inverse_transform(test21)
input2=test
if n>1:
  for i in range(1,n):
    input2=input2.reshape(len(input2),8)
    input2=DataFrame(input2)
    testresult=DataFrame(testresult)
    input2=concat([input2,testresult],axis=1)
    input2=DataFrame(input2)
    input2=input2.values
    input2=input2[:,-8:]
    input2=input2.reshape(len(input2),1,8)
    testresult=model.predict(input2,verbose=0)
    testresult=testresult.reshape(len(testresult),4)
    test22=testresult
    test22=scaler1.inverse_transform(test22)
    test21=DataFrame(test21)
    test22=DataFrame(test22)
    test21=concat([test21,test22],axis=1)
test[:,0:4]=scaler1.inverse_transform(test[:,0:4])
test[:,4:8]=scaler1.inverse_transform(test[:,4:8])
testresult=scaler1.inverse_transform(testresult)
test1=test[:,0]
test2=test[:,1]
testresult1=testresult[:,0]
testresult2=testresult[:,1]
#prediction on typhoon case
series2=read_csv('201513pressure--.csv',header=None,index_col=0)
values2=series2.values
scaler2=MinMaxScaler(feature_range=(0.1,0.9))
scaler2=scaler2.fit(values2)
normalized2=scaler2.transform(values2)
values21=DataFrame(normalized2)
values2=concat([values21,values21.shift(-1)],axis=1)
testnew1=values2.values
testnew1=testnew1[:len(testnew1)-1,:]
testnew1=testnew1.reshape(testnew1.shape[0],1,testnew1.shape[1])
testresultnew1=model.predict(testnew1,verbose=0)
testnew1=testnew1.reshape(testnew1.shape[0],testnew1.shape[2])
test31=testresultnew1
test31=scaler2.inverse_transform(test31)
input3=testnew1
if n>1:
  for i in range(1,n):
    input3=input3.reshape(len(input3),8)
    input3=DataFrame(input3)
    testresultnew1=DataFrame(testresultnew1)
    input3=concat([input3,testresultnew1],axis=1)
    input3=DataFrame(input3)
    input3=input3.values
    input3=input3[:,-8:]
    input3=input3.reshape(len(input3),1,8)
    testresultnew1=model.predict(input3,verbose=0)
    testresultnew1=testresultnew1.reshape(len(testresultnew1),4)
    test32=testresultnew1
    test32=scaler2.inverse_transform(test32)
    test31=DataFrame(test31)
    test32=DataFrame(test32)
    test31=concat([test31,test32],axis=1)
testnew1[:,0:4]=scaler2.inverse_transform(testnew1[:,0:4])
testnew1[:,4:8]=scaler2.inverse_transform(testnew1[:,4:8])
testresultnew1=scaler2.inverse_transform(testresultnew1)
testnew11=testnew1[:,0]
testnew12=testnew1[:,1]
testresultnew11=testresultnew1[:,0]
testresultnew12=testresultnew1[:,1]
test11=test11.values
test21=test21.values
test31=test31.values

#evaluation indicators calculated
measure11=read_csv('result5140.csv',header=None,index_col=None)
for i in range(1,n+1):
  mae11=mean_absolute_error(test1[i:],test21[:len(testresult1)-i,4*i-4])
  mse11=mean_squared_error(test1[i:],test21[:len(testresult1)-i,4*i-4])
  rmse11=sqrt(mean_squared_error(test1[i:],test21[:len(testresult1)-i,4*i-4]))
  mae21=mean_absolute_error(test2[i:],test21[:len(testresult2)-i,4*i-3])
  mse21=mean_squared_error(test2[i:],test21[:len(testresult2)-i,4*i-3])
  rmse21=sqrt(mean_squared_error(test2[i:],test21[:len(testresult2)-i,4*i-3]))
  mae12=mean_absolute_error(testnew11[i:],test31[:len(testresultnew11)-i,4*i-4])
  mse12=mean_squared_error(testnew11[i:],test31[:len(testresultnew11)-i,4*i-4])
  rmse12=sqrt(mean_squared_error(testnew11[i:],test31[:len(testresultnew11)-i,4*i-4]))
  mae22=mean_absolute_error(testnew12[i:],test31[:len(testresultnew12)-i,4*i-3])
  mse22=mean_squared_error(testnew12[i:],test31[:len(testresultnew12)-i,4*i-3])
  rmse22=sqrt(mean_squared_error(testnew12[i:],test31[:len(testresultnew12)-i,4*i-3]))

  mae=mean_absolute_error(observation1[i-1:],test11[:len(prediction1)-i+1,4*i-4])
  mse=mean_squared_error(observation1[i-1:],test11[:len(prediction1)-i+1,4*i-4])
  rmse=sqrt(mean_squared_error(observation1[i-1:],test11[:len(prediction1)-i+1,4*i-4]))
  mae0=mean_absolute_error(observation2[i-1:],test11[:len(prediction2)-i+1,4*i-3])
  mse0=mean_squared_error(observation2[i-1:],test11[:len(prediction2)-i+1,4*i-3])
  rmse0=sqrt(mean_squared_error(observation2[i-1:],test11[:len(prediction2)-i+1,4*i-3]))

  measure=[[0 for x in range(18)] for y in range(1)]
  measure[0][0]=mae11
  measure[0][1]=mse11
  measure[0][2]=rmse11
  measure[0][3]=mae21
  measure[0][4]=mse21
  measure[0][5]=rmse21
  measure[0][6]=mae12
  measure[0][7]=mse12
  measure[0][8]=rmse12
  measure[0][9]=mae22
  measure[0][10]=mse22
  measure[0][11]=rmse22

  measure[0][12]=mae
  measure[0][13]=mse
  measure[0][14]=rmse
  measure[0][15]=mae0
  measure[0][16]=mse0
  measure[0][17]=rmse0
  measure11=measure11.append(measure)

#avoid to the outliers
predict=test11
count=0
for i in range(len(predict)):
  for j in range(20):
    if predict[i,4*j+1]<0:
      count=count+1
print(count)
#save the prediction values and evaluation indicators
if count==0:
  data1=DataFrame(test_y)
  data2=DataFrame(predictions)
  data3=DataFrame(test)
  data4=DataFrame(testresult)
  data5=DataFrame(test11)
  data6=DataFrame(test21)
  data7=DataFrame(test31)
  """
  data20=read_csv('result5142.csv',header=None,index_col=None)
  data2=concat([data20,data2],axis=1)
  data40=read_csv('result5144.csv',header=None,index_col=None)
  data4=concat([data40,data4],axis=1)
  data50=read_csv('result5145.csv',header=None,index_col=None)
  data5=concat([data50,data5],axis=1)
  data60=read_csv('result5146.csv',header=None,index_col=None)
  data6=concat([data60,data6],axis=1)
  data70=read_csv('result5147.csv',header=None,index_col=None)
  data7=concat([data70,data7],axis=1)
  """
  data1.to_csv('testresult/514/result5141.csv',header=False,index=False)
  data2.to_csv('testresult/514/result5142.csv',header=False,index=False)
  data3.to_csv('testresult/514/result5143.csv',header=False,index=False)
  data4.to_csv('testresult/514/result5144.csv',header=False,index=False)
  data5.to_csv('testresult/514/result5145.csv',header=False,index=False)
  data6.to_csv('testresult/514/result5146.csv',header=False,index=False)
  data7.to_csv('testresult/514/result5147.csv',header=False,index=False)

  data0=DataFrame(measure11)
  data0.to_csv('testresult/514/result5140.csv',header=False,index=False)
