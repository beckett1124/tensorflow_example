#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import datasets
from sklearn import decomposition

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from mpl_toolkits.mplot3d import Axes3D
#matplotlib notebook


mnist = datasets.load_digits()
X = mnist.data
Y = mnist.target
pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(X)

