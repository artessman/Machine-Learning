import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=2)

#data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 4, 6])

# line spec for plotting
c = ['r', 'b', 'g', 'purple', 'black']
labels = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']

#for plotting
xline = np.linspace(0, 6, 10)  


# Ref slide 395 of python for data sci slides (least squares)
for i in range(len(x)):
    
    #calculate current mean
    currMeanX = np.mean(x[:i+ 1])
    currMeanY = np.mean(y[:i+ 1])

    #calculate mean of X*Y
    meanXY = np.mean(x[:i+ 1] * y[:i+ 1])
    #calculate mean of X squared
    meanXSquared = np.mean(x[:i+ 1] ** 2)

    #calculate intercept and coeff
    w1 = (meanXY - (currMeanX * currMeanY)) / (meanXSquared - (currMeanX ** 2))
    w0 = currMeanY - (w1 * currMeanX)

    #make model
    ymodel = w1 * xline + w0

    #print vals
    print(f'Model {i + 1}: Slope = {w1}, Intercept: {w0}')
    
    #plt current model
    plt.plot(xline, ymodel, color=c[i], label=labels[i])


# original data points
plt.scatter(x, y, color='black', label='Data points')

# Plot labels
plt.title('Online Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
