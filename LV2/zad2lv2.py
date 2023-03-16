import numpy as np
import matplotlib.pyplot as plt 

data = np.loadtxt('data.csv',skiprows=1,delimiter=',',)
totalPeople = data.shape[0]

#a
print(totalPeople)

#b
gender=data[:,0]
height=data[:,1]
weight=data[:,2]
#print(gender)
#print(height)
#print(weight)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(height, weight, s=0.7)
plt.show ()

#c
gender=data[0::50,0]
height=data[::50,1]
weight=data[::50,2]
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(height, weight, s=0.7)
plt.show ()

#d
print("Minimum height:", min(height))
print("Maximum height", max(height))
print("Mean height", height.mean())

#e

men_index = (data[:,0] == 1)
women_index = (data[:,0] == 0)

men_details = data[men_index]
women_details = data[women_index]

men_height = men_details[:,1]
women_height=women_details[:,1]

#men data
print(min(men_height))
print(max(men_height))
print(men_height.mean())

#women data
print(min(women_height))
print(max(women_height))
print(women_height.mean())