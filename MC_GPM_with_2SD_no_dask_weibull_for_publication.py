# There are two code files. This one, using no multithread, runs slowly, but easier to read and obtains results identical to at least 16 digits as the multithread used for the simulations.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, weibull_min
from datetime import datetime
from math import sqrt, log, exp


def Pivot_calculation(rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
    # calculate pivots for generalized confidence interval
    return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

def transform_from_raw_to_log_mean_SD(Mean, SD):
    # CV in time scale, calculated from SD in time scale and Mean in time scale
    CV = SD/Mean
    CVsq = CV**2
    # Mean in log scale
    MeanLogScale = log(Mean/sqrt(CVsq + 1)) 
    # SD in log scale
    SDLogScale = sqrt(log((CVsq + 1)))
    return MeanLogScale, SDLogScale


# number of Monte Carlo Simulations
nMonte = 1000000

# calculate z-score for alpha = 0.05, gi
# ppf is the percent point function that is inverse of cumulative distribution function
z_score = norm.ppf(1 - 0.05 / 2)        

# the number for pivot, the notation "m" in the manuscript
nSimulForPivot = 100000-1

# choosing a seed
seed_value = 12181988

# Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later
np.random.seed(seed_value - 1)
random_numbers1_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 2)
random_numbers1_2 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 3)
random_numbers2_1 = np.random.rand(nSimulForPivot)

np.random.seed(seed_value - 4)
random_numbers2_2 = np.random.rand(nSimulForPivot)

# Weibull shape and scale parameters for coresponding CV
weibull_parameters = {'MeanTimeScale':[1.0, 1.0, 1.0],
                      'CV':[0.15, 0.3, 0.5],
                      'shape_parameter':[7.906924265775673, 3.713772390158953, 2.1013488127891664],
                      'scale_parameter': [1.062466408811014,1.1078638656698665,1.1290632634554316]}
df_weibull_params = pd.DataFrame(weibull_parameters)

# Mean in time scale, we choose 0.25, 1, and 3
for MeanTimeScale in [1]:
    # Sample size, we choose 15, 25, and 50
    for N in [15, 25, 50]:
        N1 = N
        N2 = N1

        for CV in [0.15, 0.3, 0.5]:
            # coefficient of variation, we choose 0.15, 0.3, 0.5
            CV1 = CV
            CV2 = CV1

            # Weibull shape and scale parameters having the specified MeanTimeScale and CV
            shape_parameter, scale_parameter = df_weibull_params[ (df_weibull_params['MeanTimeScale'] == MeanTimeScale) & (df_weibull_params['CV'] == CV) ][['shape_parameter','scale_parameter']].iloc[0,:]

            # generate random number for later used in calculating Ui and Zi in generalized pivotal method
            # group 1 pivot calculation
            U1 = chi2.ppf(random_numbers1_1, N1 - 1 )
            Z1 = norm.ppf(random_numbers2_1)

            # group 2 pivot calculation
            U2 = chi2.ppf(random_numbers1_2, N2 - 1 )
            Z2 = norm.ppf(random_numbers2_2)

            #collecting results
            dict_results = {'ln_ratio': [], 'se_ln_ratio': [], 'coverage': []}
            # the pre-determined list of seeds, using number of nMonte
            list_seeds = [i for i in range(seed_value, seed_value + nMonte)] 
            for seed_ in list_seeds:
                # generate Weibull distribution, using mean of rMeanLogScale and standard deviation of rSDLogScale
                rSampleOfRandoms = weibull_min.rvs(shape_parameter, scale=scale_parameter, size=N1+N2, random_state = seed_)

                # split samples into group 1 and group 2
                rSampleOfRandoms1 = rSampleOfRandoms[:N1]
                rSampleOfRandoms2 = rSampleOfRandoms[N1:N1+N2] 

                rSampleMeanTimeScale1 = np.mean(rSampleOfRandoms1)
                rSampleSDTimeScale1 = np.std(rSampleOfRandoms1, ddof=1)
                rSampleMeanTimeScale2 = np.mean(rSampleOfRandoms2)
                rSampleSDTimeScale2 = np.std(rSampleOfRandoms2, ddof=1)

                # transform mean and SD from raw to log                 
                rSampleMeanLogScale1, rSampleSDLogScale1 = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale1, rSampleSDTimeScale1)
                rSampleMeanLogScale2, rSampleSDLogScale2 = transform_from_raw_to_log_mean_SD(rSampleMeanTimeScale2, rSampleSDTimeScale2)
        
                # calculate pivots
                Pivot1 = Pivot_calculation(rSampleMeanLogScale1, rSampleSDLogScale1, N1, U1, Z1)
                Pivot2 = Pivot_calculation(rSampleMeanLogScale2, rSampleSDLogScale2, N2, U2, Z2)
                
                # generalized pivotal statistics
                pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

                # calculate ln ratio and SE ln ratio by percentile and Z statistics
                ln_ratio = pd.Series(pivot_statistics).quantile(.5)
                se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
                # calculate the 95% confidence intervals with z_score
                lower_bound = ln_ratio - z_score * se_ln_ratio
                upper_bound = ln_ratio + z_score * se_ln_ratio   
                
                dict_results['ln_ratio'].append(ln_ratio)
                dict_results['se_ln_ratio'].append(se_ln_ratio)
                dict_results['coverage'].append((lower_bound < 0) and (upper_bound > 0))
            
            end_time = datetime.now()
            
            # print out the percentage of coverage
            print(f'N={N1} CV={CV1} percentage coverage: {np.mean(dict_results["coverage"])}') 
            
            output_dir = f"Weibull_GPMMC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_mean_{MeanTimeScale}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
            print(f'csv save to {output_dir}.csv')

            # save the results to the csv}
            pd.DataFrame(dict_results).to_csv(f'{output_dir}.csv')
quit()

