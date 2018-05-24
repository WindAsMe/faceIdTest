# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :18-5-18 下午8:06
# File     :PCATest.py
# Location:/Home/PycharmProjects/..

# Principal Components Analysis(PCA)


import numpy as np
from sklearn import decomposition


def do_pca():

    # Define feature
    # The last three feature is
    # DEPEND on x1, x2
    x1 = np.random.normal(size=250)
    x2 = np.random.normal(size=250)
    x3 = 2 * x1 + 3 * x2
    x4 = 4 * x1 - x2
    x5 = x3 + 2 * x4

    # Create the collection of these statics
    x = np.c_[x1, x3, x2, x5, x4]

    # Create a PCA module
    pca = decomposition.PCA()

    # DO PCA
    pca.fit(x)

    # Print the VARIANCE
    variances = pca.explained_variance_
    print('Variances is decreasing order:\n', variances)

    # Find out the useful dimensions
    thresh_variance = 0.8
    num_userful_dims = len(np.where(variances > thresh_variance)[0])
    print('\nNumber of useful dimensions:', num_userful_dims)

    # Only two dimension is useful
    pca.n_components = num_userful_dims

    x_new = pca.fit_transform(x)
    print('\nShape before:', x.shape)
    print('Shape after:', x_new.shape)


<<<<<<< HEAD
=======

>>>>>>> 8b45d1414bddcec00e1e2edf7d5c3923c08c86bb
if __name__ == '__main__':
    do_pca()
