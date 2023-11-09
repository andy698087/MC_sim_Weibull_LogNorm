# This one, using the multithread, runs quickly (about 10 min for each 100,000 simulations). It obtains results identical to at least 16 digits as the one using no multithread.
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2, weibull_min
from datetime import datetime
from math import sqrt, log, exp, gamma
import dask.dataframe as dd
from scipy.optimize import minimize

class SimulPivotMC(object):
    def __init__(self, nMonteSim, N, CVTimeScale, MeanTimeScale):
        # number of Monte Carlo Simulation
        self.nMonte = nMonteSim

        # Calculate z-score for alpha = 0.05, ppf is the percent point function that is inverse of cumulative distribution function
        self.z_score = norm.ppf(1 - 0.05 / 2)

        # Sample size, we choose 15, 25, 50, notation "n" in the manuscript
        self.N1 = N
        self.N2 = self.N1

        # coefficient of variation, we choose 0.15, 0.3, 0.5
        self.CV1 = CVTimeScale
        self.CV2 = self.CV1
        
        path_to_weibull_parameters_found = 'weibull_params1e-6_mean_025.csv'
        # Load files for Weibull shape and scale parameters for coresponding mean and CV
        df_weibull_params = pd.read_csv(path_to_weibull_parameters_found)[['MeanTimeScale','CV','shape_parameter','scale_parameter']]
        
        # Weibull shape and scale parameters having the specified MeanTimeScale and CV
        self.shape_parameter, self.scale_parameter = df_weibull_params[ (df_weibull_params['MeanTimeScale'] == MeanTimeScale) & (df_weibull_params['CV'] == CV) ][['shape_parameter','scale_parameter']].iloc[0,:]
        print(f'CVTimeScale: {CVTimeScale}')
        print(f'self.shape_parameter, self.scale_parameter: {self.shape_parameter, self.scale_parameter}')

        # the number for pivot
        nSimulForPivot = 100000-1
        
        # choosing a seed
        self.seed_value = 12181988

        # Generate 4 set of random numbers each with specified seed, will be used for U1, U2, Z1, and Z2 later
        np.random.seed(self.seed_value - 1)
        random_numbers1_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 2)
        random_numbers1_2 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 3)
        random_numbers2_1 = np.random.rand(nSimulForPivot)

        np.random.seed(self.seed_value - 4)
        random_numbers2_2 = np.random.rand(nSimulForPivot)

        # Generate random number for later used in calculating Ui and Zi in generalized pivotal method
        # group 1 pivot calculation
        self.U1 = chi2.ppf(random_numbers1_1, self.N1 - 1 )
        self.Z1 = norm.ppf(random_numbers2_1)

        # group 2 pivot calculation
        self.U2 = chi2.ppf(random_numbers1_2, self.N2 - 1 )
        self.Z2 = norm.ppf(random_numbers2_2)

    def main(self):
        # the pre-determined list of seeds, using number of nMonte
        list_seeds = [i for i in range(self.seed_value, self.seed_value + self.nMonte)] 
        # put the list of seeds into a table (a.k.a DataFrame) with one column named "Seeds"  
        print('Sample_Weibull')
        df = pd.DataFrame({'Seeds':list_seeds}) 
        df['rSampleOfRandomsWeibull'] = df.apply(self.Sample_Weibull, args=('Seeds',), axis=1)
        df_record = df
        
        # put the table into dask, a progress that can parallel calculating each rows using multi-thread
        df = dd.from_pandas(df['rSampleOfRandomsWeibull'], npartitions=35) 
        meta = ('float64', 'float64')

        print('Mean_SD')
        # calculate sample mean and SD using Mean_SD
        df = df.apply(self.Mean_SD, meta=meta)
        df_record[['WeibullMean1', 'WeibullSD1', 'WeibullMean2', 'WeibullSD2']] = df.compute().tolist()
    
        print('first_two_moment')
        # using first two moments to transform mean and SD from raw to log
        df = df.apply(self.first_two_moment, args=(0,1,2,3), meta=meta) 
        df_record[['rSampleMeanLogScale1', 'rSampleSDLogScale1', 'rSampleMeanLogScale2', 'rSampleSDLogScale2']] = df.compute().tolist()
        
        print('GPM_log_ratio_SD')
        # generate 'ln_ratio' and 'se_ln_ratio' with log mean and log SD using GPM
        df = df.apply(self.GPM_log_ratio_SD, args=(0,1,2,3), meta=meta)  
        df_record[['ln_ratio', 'se_ln_ratio']] = df.compute().tolist()
        
        print('Coverage')
        # check coverage of each rows
        df = df.apply(self.Coverage, args=(0,1), meta=meta) 
        df_record['intervals_include_zero'] = df.compute().tolist()
        print('compute dask')
        # compute the mean of the list of coverage (0 or 1), it equals to the percentage of coverage in Table
        coverage = df.mean().compute() 

        return coverage, df_record, self.nMonte, self.N1, self.CV1


    def Sample_Weibull(self, row, seed_):
        rSampleOfRandoms = weibull_min.rvs(self.shape_parameter, scale=self.scale_parameter, size=self.N1+self.N2, random_state = row[seed_])

        return rSampleOfRandoms

    def Mean_SD(self, row):

        rSampleOfRandoms1 = row[:self.N1]
        rSampleOfRandoms2 = row[self.N1:(self.N1+self.N2)]
        
        # the mean of rSampleOfRandoms1
        rSampleMean1 = np.mean(rSampleOfRandoms1)  
        # the standard deviation of rSampleOfRandoms1, delta degree of freeden = 1
        rSampleSD1 = np.std(rSampleOfRandoms1, ddof=1) 
        rSampleMean2 = np.mean(rSampleOfRandoms2)
        rSampleSD2 = np.std(rSampleOfRandoms2, ddof=1)

        return rSampleMean1, rSampleSD1, rSampleMean2, rSampleSD2
              
    def first_two_moment(self, row, col_SampleMean1, col_SampleSD1, col_SampleMean2, col_SampleSD2):
        
        SampleMean1 = row[col_SampleMean1]
        SampleSD1 = row[col_SampleSD1]

        SampleMean2 = row[col_SampleMean2]
        SampleSD2 = row[col_SampleSD2]

        # using first two moments to transform mean and SD from raw to log
        rSampleMeanLogScale1, rSampleSDLogScale1 = self.transform_from_raw_to_log_mean_SD(SampleMean1, SampleSD1)
        rSampleMeanLogScale2, rSampleSDLogScale2 = self.transform_from_raw_to_log_mean_SD(SampleMean2, SampleSD2)

        return rSampleMeanLogScale1, rSampleSDLogScale1, rSampleMeanLogScale2, rSampleSDLogScale2

    def transform_from_raw_to_log_mean_SD(self, Mean, SD):
        # CV in time scale, calculated from SD in time scale and Mean in time scale
        CV = SD/Mean
        CVsq = CV**2
        # Mean in log scale
        MeanLogScale = log(Mean/sqrt(CVsq + 1)) 
        # SD in log scale
        SDLogScale = sqrt(log((CVsq + 1)))

        return MeanLogScale, SDLogScale
    
    def GPM_log_ratio_SD(self, row, col_SampleMeanLog1, col_SampleSDLog1, col_SampleMeanLog2, col_SampleSDLog2):  # Equation 2 and 3 

        #group 1 pivot calculation
        SampleMeanLog1 = row[col_SampleMeanLog1]
        SampleSDLog1 = row[col_SampleSDLog1]
        Pivot1 = self.Pivot_calculation(SampleMeanLog1, SampleSDLog1, self.N1, self.U1, self.Z1)

        #group 2 pivot calculation
        SampleMeanLog2 = row[col_SampleMeanLog2]
        SampleSDLog2 = row[col_SampleSDLog2]
        Pivot2 = self.Pivot_calculation(SampleMeanLog2, SampleSDLog2, self.N2, self.U2, self.Z2)

        # generalized pivotal statistics
        pivot_statistics = np.log(Pivot1) - np.log(Pivot2)

        # calculate ln ratio and SE ln ratio by percentile and Z statistics
        ln_ratio = pd.Series(pivot_statistics).quantile(.5)
        se_ln_ratio = (pd.Series(pivot_statistics).quantile(.75) - pd.Series(pivot_statistics).quantile(.25))/(norm.ppf(.75) - norm.ppf(.25))
        
        return ln_ratio, se_ln_ratio
     
    def Pivot_calculation(self, rSampleMeanLogScale, rSampleSDLogScale, N, U, Z):
        # calculate pivots for generalized confidence interval
        return np.exp(rSampleMeanLogScale- np.sqrt((rSampleSDLogScale**2 * (N-1))/U) * (Z/sqrt(N)) ) * np.sqrt((np.exp((rSampleSDLogScale**2 * (N-1))/U) - 1) * np.exp((rSampleSDLogScale**2 * (N-1))/U))

    def Coverage(self, row, col_ln_ratio, col_se_ln_ratio):
        
        ln_ratio = row[col_ln_ratio]
        se_ln_ratio = row[col_se_ln_ratio]

        # calculate the 95% confidence intervals with z_score of alpha = 0.05
        lower_bound = ln_ratio - self.z_score * se_ln_ratio
        upper_bound = ln_ratio + self.z_score * se_ln_ratio   

        intervals_include_zero = (lower_bound < 0) and (upper_bound > 0)
        # 1 as True, 0 as False, check coverage
        return int(intervals_include_zero)  

        
if __name__ == '__main__':
    # number of Monte Carlo simulations
    nMonteSim = 1000000
    # Mean in time scale, we choose 0.25, 1, and 3
    for MeanTimeScale in [0.25, 1, 3]:
        # Sample size, we choose 15, 25, 50
        for N in [15, 25, 50]: 
            # coefficient of variation, we choose 0.15, 0.3, 0.5
            for CV in [0.15, 0.3, 0.5]: 
                # record the datetime at the start
                start_time = datetime.now() 
                print('start_time:', start_time) 
                print(f"Start GPM_MC_nMonteSim_{nMonteSim}_N_{N}_CV_{CV}_mean_{MeanTimeScale}_{str(start_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}")

                # Cal the class SimulPivotMC(), generate variables in the def __init__(self)
                run = SimulPivotMC(nMonteSim, N, CV, MeanTimeScale)  
                # start main()
                coverage_by_ln_ratio, df_record, nMonte, N1, CV1 = run.main()  
                
                # record the datetime at the end
                end_time = datetime.now() 
                # print the datetime at the end
                print('end_time:', end_time) 
                # calculate the time taken
                time_difference = end_time - start_time
                print('time_difference:', time_difference) 
                # print out the percentage of coverage
                print('percentage coverage: %s' %(coverage_by_ln_ratio,)) 
                    
                output_txt = f"start_time: {start_time}\nend_time: {end_time}\ntime_difference: {time_difference}\n\nnMonte = {nMonte}; N1 = {N1}; CV1 = {CV1}\n\n percentage coverage: {coverage_by_ln_ratio}\n"
                
                output_dir = f"Weibull_GPMMC_nMonte_{nMonte}_N_{N1}_CV_{CV1}_mean_{MeanTimeScale}_{str(end_time).split('.')[0].replace('-','').replace(' ','').replace(':','')}"
                
                # save the results to the csv
                print(f'csv save to {output_dir}.csv')
                df_record.to_csv(f'{output_dir}.csv')
                
                # save the results to the txt
                with open(f'{output_dir}.txt', 'w') as f:
                    f.write(output_txt)
quit() 