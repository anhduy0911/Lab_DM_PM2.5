from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from yaml import full_load

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

class ForestRegressorModel:
    def __init__(self):
        self.data = self.__process_data()
        self.model_param = full_load(open('Parameter Configure/Forest Param/forest_param.yaml'))
        self.model = RandomForestRegressor(**self.model_param)
    
    def __process_data(self):
        data = pd.read_csv('Processing Data/processed_data.csv',)
        data = data.drop(['date_time','wind_dir','temp'],axis = 1).astype(float)
        data = data.replace([np.inf,-np.inf],np.nan)
        data = data.dropna(axis=0,how='any')

        X = np.array(data.iloc[:,0:3])
        y = np.array(data['pm_2.5'])
        print(X[:5])
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
        #print((X_train.shape,y_train.shape,X_test.shape,y_test.shape))
        return (X_train,y_train,X_test,y_test)

    def train_model(self):
        self.model.fit(self.data[0],self.data[1])
    
    def evaluate_and_plot(self):
        y_predict = self.model.predict(self.data[2])
        #print((self.data[3].shape,y_predict.shape))
        score = r2_score(self.data[3],y_predict)
        print('Evaluation of model: ',score)
        print((self.data[3][:10],y_predict[:10]))
        plt.plot(range(self.data[3].shape[0]),self.data[3],'or',label='y_test')
        plt.plot(range(self.data[3].shape[0]),y_predict,'ob',label='predict_val')
        plt.ylabel('PM 2.5')
        plt.legend()
        plt.savefig('Log/Random Forest/comparation.png')
        plt.show()

if __name__ == '__main__':
    model = ForestRegressorModel()
    model.train_model()
    model.evaluate_and_plot()

