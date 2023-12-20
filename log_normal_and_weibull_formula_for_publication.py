# This is the code for finding Weibull two parameters having population mean in time scale and variance in time scale approximate the log-normal distributions with difference below 1E-6 hours
import numpy as np
import pandas as pd
from math import gamma
from scipy.optimize import minimize
from time import sleep
from datetime import datetime

class weibull_and_lognorm(object):

    def __init__(self, MeanTimeScale, CVTimeScale, seed_=20230922):
        # set the MeanTimeScale and calculate the VarTimeScale according to MeanTimeScale and CVTimeScale
        self.MeanTimeScale = MeanTimeScale
        self.VarTimeScale = (CVTimeScale * self.MeanTimeScale)**2

        print('self.MeanTimeScale:',self.MeanTimeScale)
        print('self.VarTimeScale:',self.VarTimeScale)
        sleep(2)

        # set a seed for reproducibility
        self.seed_value = seed_
        
        # start point x0 for finding the Weibull parameters
        self.x0_pre = [10,1]

    def find_WeibullMeanVar(self):
        # count the iteration with nIter
        nIter = 0
        # set the start point x0
        x0 = self.x0_pre
        # build a dictionary to keep each parameters
        self.dict_WeibullParameter_diff = {'MeanTimeScale': [], 'VarTimeScale': [], 'shape_parameter': [], 'scale_parameter': [], 'MeanWeibull': [], 'VarWeibull': [], 'diff': [], 'diff_mean': [], 'diff_var': []}
        # set the boundaries for finder
        bounds_ = [(0.5,None),(0.1,None)]
        # set the options for finder
        # ftol is the precision goal for the value of function in the stopping criterion
        # xtol is the precision goal for the value of x in the stopping criterion
        # eta is severity of the line search, I set it to be decrease while the iteration number increase
        # display will show the detail
        options_ = {'ftol': 1e-8, 'xtol': 1e-10, 'eta': 0.01/(nIter//1000 + 1), 'disp': False}

        # make an eternal loop until condition satisfied
        while True:
            # count the iteration
            nIter += 1
            # the output of the function "sum_diff_weibull_lognorm" will be minimized, using the arguments of MeanTimeScale and VarTimeScale
            # the algorithm 'TNC' is used, with  3-point numerical approximation of the jacobian
            res = minimize(self.sum_diff_weibull_lognorm, x0, args=(self.MeanTimeScale, self.VarTimeScale), method='TNC', bounds=bounds_, options=options_, jac = '3-point')
            # if the dict_WeibullParameter_diff had values recorded, check the difference
            if len(self.dict_WeibullParameter_diff['diff']) > 0:
                extract_df = pd.DataFrame(self.dict_WeibullParameter_diff)
                extract_df = extract_df.loc[extract_df['diff'].idxmin()]                
                diff = extract_df['diff']
                # if the difference below 1e-6, get the parameter from result
                if diff < 1e-6:
                    self.extract_df = extract_df
                    shape_parameter, scale_parameter = res.x 
                    print('extract_df', extract_df)        
                    break
            else:
                # set a new start x0 for a new search with a random shift
                np.random.seed(self.seed_value+nIter)
                random_ = (1 + (np.random.randint(1,high=10)-5)/100)
                x0 = [max(0.5, res.x[0] * random_), max(0.5, res.x[1] * random_) ]

        return shape_parameter, scale_parameter
    
    # define the function with the result to be minimized
    def sum_diff_weibull_lognorm(self, params, MeanTimeScale, VarTimeScale):
        # make sure the input parameters are all above or equal to zero
        shape_parameter, scale_parameter = abs(params)
        # calculate the MeanWeibul and VarWeibull according to the formula defined by formula_Weibull_MeanVar
        MeanWeibull, VarWeibull = self.formula_Weibull_MeanVar(shape_parameter, scale_parameter)
        # calculate the difference of Mean and Variance between Weibull and LogNorm
        diff = abs(MeanWeibull - MeanTimeScale) + abs(VarWeibull - VarTimeScale)
        diff_mean = abs(MeanWeibull - MeanTimeScale)
        diff_var = abs(VarWeibull - VarTimeScale)
        # if the difference of each are below 1e-6, record the shape and scale parameters
        if diff < 1e-6 and diff_mean < 1e-6 and diff_var < 1e-6:
            self.dict_WeibullParameter_diff['MeanTimeScale'].append(MeanTimeScale)
            self.dict_WeibullParameter_diff['VarTimeScale'].append(VarTimeScale)
            self.dict_WeibullParameter_diff['shape_parameter'].append(shape_parameter)
            self.dict_WeibullParameter_diff['scale_parameter'].append(scale_parameter)
            self.dict_WeibullParameter_diff['MeanWeibull'].append(MeanWeibull)
            self.dict_WeibullParameter_diff['VarWeibull'].append(VarWeibull)
            self.dict_WeibullParameter_diff['diff'].append(diff)
            self.dict_WeibullParameter_diff['diff_mean'].append(diff_mean)
            self.dict_WeibullParameter_diff['diff_var'].append(diff_var)
        
        return self.loss_func(MeanWeibull, VarWeibull, MeanTimeScale, VarTimeScale)     
    
    # calculate the weibull mean and variance using shape and scale parameters according to the formula
    def formula_Weibull_MeanVar(self, shape_parameter, scale_parameter):
        MeanWeibull = scale_parameter * gamma(1 + 1/shape_parameter)
        VarWeibull = (scale_parameter ** 2) * (gamma(1 + 2/shape_parameter) - (gamma(1 + 1/shape_parameter)) ** 2)

        return MeanWeibull, VarWeibull
    # define the loss function to calculate the difference mean and variance between between Weibull and LogNorm
    def loss_func(self, a1, a2, b1, b2):
        return abs(a1-b1)+ abs(a2-b2)
    
# collect the parameters
weibull_params = {'mean':[],'CV':[],'shape_parameter':[],'scale_parameter':[],'MeanTimeScale': [], 'VarTimeScale': [], 'shape_parameter': [], 'scale_parameter': [], 'MeanWeibull': [], 'VarWeibull': [], 'diff': [], 'diff_mean': [], 'diff_var': []}
# coefficient of variation, we choose 0.15, 0.3, and 0.5
for CV in [0.15, 0.30, 0.5]:
    # mean in time scale, we choose 0.25, 1, and 3
    for mean in [1]:
        print(f'mean, CV: {mean}, {CV}')
        weibull_params['mean'].append(mean)
        weibull_params['CV'].append(CV)
        fun = weibull_and_lognorm(mean, CV)
        shape_parameter, scale_parameter = fun.find_WeibullMeanVar()
        weibull_params['shape_parameter'].append(shape_parameter)
        weibull_params['scale_parameter'].append(scale_parameter)
        extract_df = getattr(fun, 'extract_df')
        weibull_params['MeanTimeScale'].append(extract_df['MeanTimeScale'])
        weibull_params['VarTimeScale'].append(extract_df['VarTimeScale'])
        weibull_params['MeanWeibull'].append(extract_df['MeanWeibull'])
        weibull_params['VarWeibull'].append(extract_df['VarWeibull'])
        weibull_params['diff'].append(extract_df['diff'])
        weibull_params['diff_mean'].append(extract_df['diff_mean'])
        weibull_params['diff_var'].append(extract_df['diff_var'])
# save the found parameters
pd.DataFrame(weibull_params).to_csv('to_your_path.csv')
quit()
