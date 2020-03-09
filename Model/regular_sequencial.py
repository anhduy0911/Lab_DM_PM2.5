from keras.models import Model
from keras.layers import Dense,Dropout,Input
from keras import optimizers
from keras.utils import normalize
from keras.callbacks.callbacks import EarlyStopping,ModelCheckpoint
#from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import KFold, cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 



class SequencialModel:

    def __init__(self,csv_path):
        self.path = csv_path
        self.model = self.build_model()
        self.data = self.__fetch_data()

    def __fetch_data(self):
        #load data
        data = pd.read_csv(self.path)
        data = data.drop(['date_time','wind_dir','temp'],axis = 1).astype(float)

        x_train = np.array(data.iloc[:30000,0:3])
        print(x_train[0])
        minmax_scaler_x = MinMaxScaler(feature_range=(0,1))
        minmax_scaler_y = MinMaxScaler(feature_range=(0,1))

        x_train_normalized = minmax_scaler_x.fit_transform(x_train)
        y = np.array(data[data['pm_2.5'] < 100].iloc[:30000,3])
        y_train = minmax_scaler_y.fit_transform(y.reshape(-1,1))

        x_val = minmax_scaler_x.transform(np.array(data.iloc[30000:32000,0:3])) 
        np.nan_to_num(data.iloc[30000:32000,3],copy=False)
        y_val = minmax_scaler_y.transform(np.array(data.iloc[30000:32000,3]).reshape(-1,1))

        x_test = minmax_scaler_x.transform(np.array(data.iloc[32000:,0:3])) 
        y_test = minmax_scaler_y.transform(np.array(data.iloc[32000:,3]).reshape(-1,1))

        return (x_train_normalized,y_train,x_val,y_val,x_test,y_test)

    def build_model(self):
        input = Input(shape=(3,))
        layer_1 = Dense(64,activation='relu')(input)
        layer_2 = Dropout(0.5)(layer_1)
        #layer_3 = Dense(128,activation='relu')(layer_2)
        #layer_4 = Dropout(0.5)(layer_3)
        output = Dense(1)(layer_2)
        model = Model(inputs=input,outputs=output,dtype=float)
        #configure model
        sgd = optimizers.SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
        model.compile(optimizer = sgd,loss='mse',metrics=['accuracy'])
        return model

    def train_model(self):
        #callbacks
        es = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
        check_point = ModelCheckpoint('Log/best_sequencial.h5',monitor='val_loss',save_best_only=True,mode='min')
        #train
        hist = self.model.fit(self.data[0],self.data[1],validation_data=(self.data[2],self.data[3]),batch_size=128,epochs=50,callbacks=[es,check_point])
        print(self.model.summary())
        return hist

    def predict_and_plot(self,hist):
        #evaluate and plot
        y_predict = self.model.predict(self.data[4])
        evals = self.model.evaluate(self.data[4],self.data[5])
        print(evals)

        figure = plt.figure(figsize=(10,5))
        figure.add_subplot(121)
        plt.plot(hist.history['loss'],'r-',label='train')
        plt.plot(hist.history['val_loss'],'b-',label='validate')
        plt.legend(loc='best')
        plt.title('Loss of train set and validate set')

        figure.add_subplot(122)
        plt.plot(range(729),self.data[5],'r-',label='GT')
        plt.plot(range(729),y_predict,'b-',label='Test')
        plt.legend(loc='best')
        plt.title('Result train set and test set')
        plt.savefig('Log/sequencial.png')
        plt.show()


if __name__ == '__main__':
    seq = SequencialModel('Processing Data/processed_data.csv')
    # estimator = KerasRegressor(build_fn=seq.build_model,epochs=50,batch_size=5,verbose= 0)
    # k_fold = KFold(n_splits=10,shuffle=True)
    # result = cross_val_score(estimator,seq.data[0],seq.data[1],cv=k_fold)
    # print("Baseline: %.2f (%.2f) MSE" % (result.mean(), result.std()))
    hist = seq.train_model()
    seq.predict_and_plot(hist)












