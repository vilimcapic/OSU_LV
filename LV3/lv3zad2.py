import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\User\Desktop\Osnove strojnog ucenja\LV3\data_C02_emission.csv')

#a

# plt.figure()
# data['CO2 Emissions (g/km)'].plot(kind='hist' )
# plt.show()

#b
#https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category

# fuelGroups = data.groupby('Fuel Type')
# fig, ax = plt.subplots()
# for name,group in fuelGroups:
#     ax.scatter(group['Fuel Consumption City (L/100km)'], group['CO2 Emissions (g/km)'])
# plt.show()


#c

# fuelGroups = data.groupby('Fuel Type')
# fuelGroups.boxplot(column='Fuel Consumption Hwy (L/100km)')
# plt.show()


#d

# fuelGroups = data.groupby('Fuel Type').agg(Size = ('Fuel Type','size'))
# fuelGroups.plot(kind='bar')
# plt.show()

#e

# cylinderGroups = data.groupby('Cylinders').agg(Avg_CO2 = ('CO2 Emissions (g/km)','mean'))
# cylinderGroups.plot(kind='bar')
# plt.show()
