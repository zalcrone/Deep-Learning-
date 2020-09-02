#importing the libraries
import numpy as np
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
p = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#scaling the dataset
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
p = sc.fit_transform(p)

#intializing with random weights
""" ##This model searches out for potentailly fraud customers.
    The Green boxes shows that these group of customers got recommendation and the red circle shows that they didn't got approval.## """
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(p)
som.train_random(data = p, num_iteration = 100)


#plotting the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(p):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2) 
show()

#printing ids of fraud customers
mappings = som.win_map(p)
frauds = np.concatenate((mappings[(7,8)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)

print("FRAUD CUSTOMER IDs")
for i in frauds[:, 0]:
    print(int(i))
    


