import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

data = pd.read_csv('Processing Data/processed_data.csv')
f_data = data.iloc[:,1:7].astype(float)
correlations = f_data.iloc[:10000,:].corr() #correlation matrix between the aspect

# figure = plt.figure(figsize=(5,5))
# plt.pcolor(correlations)
# plt.colorbar()
# plt.title('Correlation')
# plt.savefig('Processing Data/correlation.png')
# plt.show()

#from the correlation map, pm2.5 closely related to pm10 and wind speed, rh


figure = plt.figure(figsize=(10,5))
figure.add_subplot(131)
plt.plot(data['pm_10'][:10000],data['pm_2.5'][:10000],'og')
plt.title('PM10 vs PM2.5')
figure.add_subplot(132)
plt.plot(data['wind_speed'][:10000],data['pm_2.5'][:10000],'ob')
plt.title('Wind Speed vs PM2.5')
figure.add_subplot(133)
plt.plot(data['rh'][:10000],data['pm_2.5'][:10000],'or')
plt.title('RH vs PM2.5')
plt.savefig('Processing Data/linear_plot.png')
plt.show()
