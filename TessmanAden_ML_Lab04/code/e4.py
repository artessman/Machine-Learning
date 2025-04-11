
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

california_housing = fetch_california_housing(as_frame=True)

df = california_housing.frame
df2= df[::10]

X = np.array(df2.drop(['MedHouseVal'], axis=1))
y= np.array(df2['MedHouseVal'])

#exercise 4

data = pd.DataFrame(df2.drop(['Longitude', 'Latitude'], axis=1))
data_x = pd.DataFrame(df2.drop(['Longitude', 'Latitude', 'MedHouseVal'], axis=1))
sns.pairplot(data=data, vars=data_x, hue='MedHouseVal')
plt.show()
