import pandas as pd

num_of_attrs = 14
filter_mask = [12,11,0,8,6,7]

#read from file raw
path = 'Raw Data/raw_taiwan_data.csv'

raw_data = pd.read_csv(path)


# write to new csv file
processed_data = pd.DataFrame(columns=['date_time','wind_speed','wind_dir','temp','rh','pm_10','pm_2.5'])

for i in range(365*3+1,365*3+273):
    for j in range(24):
        row = {}
        row['date_time'] = raw_data['time'][i*num_of_attrs] + ' {}h'.format(j)
        row['wind_speed'] = raw_data.iloc[i*num_of_attrs + filter_mask[0],2+j]
        row['wind_dir'] = raw_data.iloc[i*num_of_attrs + filter_mask[1],2+j]
        row['temp'] = raw_data.iloc[i*num_of_attrs + filter_mask[2],2+j]
        row['rh'] = raw_data.iloc[i*num_of_attrs + filter_mask[3],2+j]
        row['pm_10'] = raw_data.iloc[i*num_of_attrs + filter_mask[4],2+j]
        row['pm_2.5'] = raw_data.iloc[i*num_of_attrs + filter_mask[5],2+j]
        processed_data = processed_data.append(row,ignore_index=True)

print(processed_data.head())
print(processed_data.tail())

processed_data.to_csv('processed_data.csv',mode='a',index =False,header=False)
    