import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import linear_model

import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import plotly.express as px
