import pandas as pd

data = pd.read_csv(r'C:\Users\User\Desktop\Osnove strojnog ucenja\LV3\data_C02_emission.csv')
# a

# data = data.astype('category')
# print(data.info())
# print(len(data))
# print(data.dtypes)
# print(data.isnull().sum())
# data.dropna(axis=0)
# data.dropna(axis=1)
# data.drop_duplicates()
# data = data.reset_index(drop=True)

# b

# sortedByFuelConsumption = data.sort_values(
# by=['Fuel Consumption City (L/100km)'])

# print(sortedByFuelConsumption[['Make', 'Model',
# 'Fuel Consumption City (L/100km)']].head(3), 'Smallest fuel consumption')

# print(sortedByFuelConsumption[['Make', 'Model',
# 'Fuel Consumption City (L/100km)']].tail(3), 'Largest fuel consumption')

# c
# midSizeEngineCars = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
# averageEmissions = midSizeEngineCars['CO2 Emissions (g/km)'].mean()
# print(len(midSizeEngineCars))
# print(averageEmissions)

#d

# audis = data[data['Make'] == 'Audi']
# audiAverageEmissions = audis[audis['Cylinders'] == 4]['CO2 Emissions (g/km)'].mean()
# print(len(audis))
# print(audiAverageEmissions)

#e

# dieselConsumptionAverage = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean()
# dieselConsumptionMedian = (data[data['Fuel Type'] == 'D'])['Fuel Consumption City (L/100km)'].median()
# print(dieselConsumptionAverage)
# print(dieselConsumptionMedian)

# petrolConsumptionAverage = (data[data['Fuel Type'] == 'X'])['Fuel Consumption City (L/100km)'].mean()
# petrolConsumptionMedian = (data[data['Fuel Type'] == 'X'])['Fuel Consumption City (L/100km)'].median()
# print(petrolConsumptionAverage)
# print(petrolConsumptionMedian) 

#g

# diesel4cylVehicles = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
# print(diesel4cylVehicles[(diesel4cylVehicles['Fuel Consumption City (L/100km)'] == diesel4cylVehicles['Fuel Consumption City (L/100km)'].max())])

#h

#print(len(data[data['Transmission'].str.startswith('M')]))

#i

print(data.corr())