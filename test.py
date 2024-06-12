
from sklearn.metrics.cluster import contingency_matrix
from seaborn import heatmap
from classes.kmeans import Kmeans
from classes.data_preparation import get_data
import matplotlib.pyplot as plt
import numpy as np

df = get_data()
