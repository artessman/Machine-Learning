
import plotly
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('materials.csv')


markercolor = data['Time']

#Make Plotly figure
fig1 = go.Scatter3d(x=data['Pressure'],
                    y=data['Temperature'],
                    z=data['Strength'],
                    marker=dict(color=markercolor,
                                opacity=1,
                                reversescale=True,
                                colorscale='Blues',
                                size=5),
                    line=dict (width=0.02),
                    mode='markers')

#Make Plot.ly Layout
mylayout = go.Layout(scene=dict(xaxis=dict( title="Pressure"),
                                yaxis=dict( title="Temperature"),
                                zaxis=dict(title="Strength")),)

#Plot and save html

plotly.offline.plot({"data": [fig1],
                     "layout": mylayout},
                     auto_open=True,
                     filename=("4DPlot.html"))