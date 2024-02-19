# -*- coding: utf-8 -*-
"""
Created on 2020-05-30

@author: Yuan Hu

"""

import numpy as np
import statsmodels.api as sm

def Compare_POOSR2(pred, true, naive, bench):
    """
    Conducts a Pseudo-Out-of-Sample R-squared test.
    
    Different prediction methods can be compared statistically based on their R-squared.

    Parameters:
    pred (array-like): Predicted values from the model being tested.
    true (array-like): Actual values.
    naive (array-like): Predictions from a naïve benchmark model.
    bench (array-like): Predictions from a fixed parameter benchmark model.

    Returns:
    statsmodels RegressionResults: The regression results instance that contains
                                   the test statistic for comparing the predictive accuracy.
    """

    # Calculate the prediction errors for the proposed model.
    e1 = pred - true
    # Calculate the prediction errors for the naïve benchmark.
    enaive = naive - true

    # Square the prediction errors for the proposed model.
    e1_2 = np.square(e1)
    # Square the prediction errors for the naïve benchmark.
    enaive_2 = np.square(enaive)

    # Calculate the pseudo R-squared for the proposed model.
    pseudo = 1 - e1_2 / np.mean(enaive_2)

    # Calculate the squared errors for the fixed parameter benchmark model.
    fix = np.square(bench - true)
    # Calculate the pseudo R-squared for the fixed parameter benchmark.
    pseudo_fix = 1 - fix / np.mean(enaive_2)

    # Perform the regression to compare the pseudo R-squared values.
    reg = sm.OLS(pseudo - pseudo_fix, np.ones(len(pseudo))).fit(cov_type='HAC', cov_kwds={'maxlags':1})

    return reg

