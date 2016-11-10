#!/usr/bin/env python3

import numpy as np
from sklearn.externals import joblib

# load the RANSAC coefficients and use them to calculate epsilon values for 'xt' Teffs
if __name__ == "__main__":
    model = joblib.load('teff_ep_RANSAC.pkl')
    xt = np.arange(5000,7000,200)
    print(model.predict(xt[:,np.newaxis]), xt)
