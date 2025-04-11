
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('vehicles.csv')
X= np.array(data.loc[:, 'cyl':])
y_mpg = np.array(data.loc[:, 'mpg'])
data = data.drop(['make', 'mpg'], axis =1) #droping colums make and mpg from data so their 
                                            #column names are not included in the data.columns that is called later

print(X)

xScaledData = StandardScaler()
xScaled = xScaledData.fit_transform(X)

reg = LinearRegression().fit(xScaled, y_mpg)#between scaledX and y

print(f'Weighted Coeff (scaled X): {reg.coef_}')
print(f'Intercept: {reg.intercept_}')

temp = abs(reg.coef_)
fiveBestColIndex = temp.argsort()[-5:]

bestX = data.iloc[:, fiveBestColIndex]
print(bestX)

markersize = data['hp']/12
markercolor = y_mpg
markershape = data['am'].map({1: 'circle', 0: 'square'})
# ['am', 'qsec', 'hp', 'disp', 'wt']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['wt'],
                    y=data['qsec'],
                    z=data['disp'],
                    marker=dict(size=markersize,
                                color=markercolor,
                                symbol=markershape,
                                opacity=0.9,
                                reversescale=True,
                                colorscale='Blues'),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="wt"),
                                yaxis=dict( title="qsec"),
                                zaxis=dict(title="disp")),)

#Plot and save html
plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("6DPlot.html"))