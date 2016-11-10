#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
#import
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
import seaborn as sns

def get_data(sfile="Data/LEGACY_teff_ep.txt"):
    k, t, te, ep, epe = np.genfromtxt(sfile).T
    n_extra = 100
    k = np.append(k, np.random.randn(n_extra))
    t = np.append(t, np.random.randn(n_extra)*200 + 5900.0)
    te = np.append(te, np.random.randn(n_extra)*50.0)
    ep = np.append(ep, np.random.randn(n_extra)*0.2+1.2)
    epe = np.append(epe, np.random.randn(n_extra)*0.01 + 0.05)
    return k, t, te, ep, epe

# use 3 different estimators to fit the Teff/epsilon data
def est(t, te, ep, epe):
    xt = np.arange(4800, 7000, 10)
    estimators = [('OLS', linear_model.LinearRegression()),
              ('Theil-Sen', linear_model.TheilSenRegressor(random_state=42)),
              ('RANSAC', linear_model.RANSACRegressor(random_state=42)), ]
    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(2), estimator)
        model.fit(t[:, np.newaxis], ep)
        y_plot = model.predict(xt[:, np.newaxis])
	# get the coefficients of the RANSAC fit to the data and put them
	# in a pickle
        if name == 'RANSAC':
            print(estimator.estimator_.coef_)
            coef = estimator.estimator_.coef_.flatten()
            joblib.dump(model, 'teff_ep_RANSAC.pkl')
            inlier_mask = estimator.inlier_mask_
            plot_(t, te, ep, epe, model=y_plot, modelx=xt, show=True, \
                  im=inlier_mask, save=name)
        else:
            print(estimator.coef_)
            plot_(t, te, ep, epe, model=y_plot, modelx=xt, show=True, \
                  save=name)
        print("Estimator : ", name)
#    plt.show()

# plot the data with the RANSAC coefficients to check it works
def plot_(t, te, ep, epe, show=True, save=[], model=[], modelx=[], im=[]):
    fig, ax = plt.subplots()

    # plot the data
    ax.plot(t, ep, 'kD')
    ax.errorbar(t, ep, fmt='kD', xerr=te, yerr=epe)

    if len(model) > 0:
        ax.plot(modelx, model, 'r-')
    if len(im) > 0:
        ax.plot(t[im], ep[im], 'rs')
    if len(save) > 0:
        plt.savefig(save + '_fit.png')
    if show:
        plt.show()

if __name__== "__main__":
    kic, teff, teff_err, ep, ep_err = get_data()
    est(teff, teff_err, ep, ep_err)
#    plot_(teff, teff_err, ep, ep_err)
