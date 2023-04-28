"""This module contains base class used for random sampling"""

import numpy as np
import pandas as pd
import scipy
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from matplotlib import gridspec
import industrial_market_simulation.support_func as imss
from sklearn.model_selection import train_test_split
#------------------------------------------------------------------

class base_consumer_sampling:
    def __init__(
        self,
        N,
        income_data,
        age_data,
        include_no_income = True, #include all age allow min income age
        sample_min_income = 6,
        min_income_age = 18, #min age to have earning
        max_income_age = 80, #min age to have earning
        male_ratio = 0.5,
        #{education level:(ratio,min_income,min_age)}
        education_lvl = {"No diploma":(0.2,None,None),
                         "High school":(0.5,None,18),
                         "Bachelor":(0.25,None,22),
                         "Master|PhD":(0.05,None,26)},
        income_correlation = {"age":0.65,
                             "education_lvl":0.75},
        other_correlation = None,
        N_modifier = 1.1,
        seed = None
        ):
        
        if N<30:
            raise ValueError("Sample size 'N' must be at least 30 to be significant.")
        #convert input array to mean, std
        #accept input type ('ratio','array','normalize')
        if income_data[0]=='ratio': # ('ratio',value_array,ratio_array)
            bins = np.array(income_data[1])
            ratio = np.array(income_data[2])
            #This need to be fixed, not correct
            income_data = bins*ratio
            income_mean = income_data.mean()
            income_std = income_data.std()
        elif income_data[0]=='array':
            income_data = np.array(income_data[1])
            income_mean = income_data.mean()
            income_std = income_data.std()
        elif income_data[0]=='normalize':
            income_mean = income_data[1]
            income_std = income_data[2]
        else:
            raise ValueError("'income_data' input method not supported.")
            
        if age_data[0]=='ratio': # (value_array,ratio_array)
            bins = np.array(age_data[1])
            ratio = np.array(age_data[2])
            #This need to be fixed, not correct
            age_data = bins*ratio
            age_mean = age_data.mean()
            age_std = age_data.std()
        elif age_data[0]=='array':
            age_data = np.array(age_data[1])
            age_mean = age_data.mean()
            age_std = age_data.std()
        elif age_data[0]=='normalize':
            age_mean = age_data[1]
            age_std = age_data[2]
        else:
            raise ValueError("'age_data' input method not supported.")
        
        self._income_data = income_data
        self._age_data = age_data
        
        np.random.seed(seed)
        
        #try to turn education level to continuously variable
        n_edu = len(education_lvl)
        p_edu = [item[1][0]for item in education_lvl.items()]
        edu_sample = np.random.choice(n_edu,size=n_edu*5000,p=p_edu)
        edu_mean = edu_sample.mean()
        edu_std = edu_sample.std()
        
        #create covariance matrix
        corr_matrix = create_corr_matrix([income_correlation['age'],
                                          income_correlation['education_lvl']],
                                         remain_corr= 0.0 if other_correlation
                                         is None else other_correlation,
                                         seed=seed)
        covar_matrix = convert_corr_to_covar(corr_matrix,
                                             np.diag(np.sqrt([income_std**2,
                                                              age_std**2,
                                                              edu_std**2])))
        multi_random_seed = seed #keep seed value for loop
        random_size = round(N_modifier*N) #increase initial random size comparing to N
        sample_buffer = []
        added_sample = 0
        error_counter = 0
        
        edu_property = [item[1] for item in education_lvl.items()]
        education_ratio = [item[1][0] for item in education_lvl.items()]
        
        #create constraint for the number of people with said education lvls
        req_edu_list,req_sample_dist = np.unique(
            np.random.choice(len(education_ratio),
                             size=N,p=education_ratio),
            return_counts=True)
        
        # keep check of current sample in each education lvl
        sample_dist = np.zeros(req_edu_list.size)
        
        #create bins
        education_ratio = np.array(education_ratio)*100
        education_ratio = np.insert(education_ratio,0,0)
        education_ratio = np.cumsum(education_ratio)
        
        #SAMPLE GENERATION
        while True:
            np.random.seed(multi_random_seed)
            sample = np.random.multivariate_normal([income_mean,age_mean,edu_mean],
                                                   covar_matrix,
                                                   size=random_size)
            #remove all negative value in the array
            sample = sample[~np.any(sample<0,axis=1)]
            if not include_no_income:
                sample = sample[sample[:,1]>=min_income_age]
            else:
                # sample = sample[sample[:,1]>=0]
                sample[(sample[:,1]<min_income_age)|
                       (sample[:,1]>max_income_age),0]=0
                # sample[(sample[:,1]<min_income_age),2]=0 # no education yet
            if not include_no_income:
                sample = sample[sample[:,0]>=sample_min_income]
            else: # include all 0 income (out of working age)
                sample = sample[(sample[:,0]==0) | 
                                (sample[:,0]>=sample_min_income)]
            #------TESTING
            edu_idx =  np.digitize(
                sample[:,2],
                bins = [np.percentile(sample[:,2],perc) if perc!=100
                        else (np.percentile(sample[:,2],perc))
                        for perc in education_ratio]
                )
            
            sample[:,2] = edu_idx-1
            dummy_list = []
            for i in req_edu_list.tolist():
                n_required = req_sample_dist[i] #current number of sample required
                n_slice = int(req_sample_dist[i] - sample_dist[i])
                if sample_dist[i]>=n_required: #pass if no more data needed
                    dummy_list.extend([None])
                    continue
                dummy_array = sample[sample[:,2]==i]
                min_income= edu_property[i][1]
                min_age = edu_property[i][2]
                if min_income is not None:
                    dummy_array = dummy_array[dummy_array[:,0]>=min_income]
                if min_age is not None:
                    dummy_array = dummy_array[dummy_array[:,1]>=min_age]
                dummy_array = dummy_array[:n_slice]
                sample_dist[i]+=len(dummy_array)# No of added new sample to dummy list
                dummy_list.append(dummy_array)
            #------
            new_sample = np.concatenate([item for item in dummy_list if item is not None ])

            #check condition for continue loop
            if np.max(req_sample_dist-sample_dist)<=0:
                sample_buffer.append(new_sample)
                break
            else:
                #max required sample 
                max_loc = np.argmax(req_sample_dist-sample_dist)  
                if len(dummy_list[max_loc])>0:
                    #new sample multifier for the next loop
                    # x random sample -> y accepted sample -> new ratio required
                    next_loop_ratio = ((req_sample_dist-sample_dist)[max_loc]/
                                       len(dummy_list[max_loc])
                                       )
                else:
                    next_loop_ratio = 5
            n = len(new_sample)
            added_sample += n
            if n==0:
                error_counter += 1
                if error_counter>=4:
                    raise ValueError("Sample couldn't be generated with the initial arguments.")
            sample_buffer.append(new_sample)
            
            #add condition for education distribution generated sample
            if added_sample<N: 
                #set random sample size for next loop
                random_size = int(np.ceil(N_modifier*next_loop_ratio*(random_size)))
                if multi_random_seed is not None:
                    multi_random_seed +=1
                continue
            else:
                break
        
        data = np.concatenate(sample_buffer)
        #add gender
        data = np.hstack((data,
                         np.random.choice(2,size=N,
                                          p=[male_ratio,1-male_ratio])
                         [:,np.newaxis])
                         )
        if include_no_income:
            n_data = len(data)
            working_status = np.zeros(n_data)
            working_status[~(data[:,0]==0)]=1
            working_status = working_status[:,np.newaxis]
            df = pd.DataFrame(data,columns=['income','age','education',
                                            'gender'])
            df['work_status']=working_status
            df.work_status.replace({0:"No",1:"Yes"},inplace=True)
        else:
            df = pd.DataFrame(data,columns=['income','age','education',
                                            'gender'])
        df.gender.replace({0:"male",1:"female"},inplace=True)
        numeric_edu = df.education.unique()
        df.education.replace(dict(zip(numeric_edu,
                                      education_lvl.keys())),
                             inplace=True)
        self.df = df
        self.education_lvl = education_lvl
        self._include_no_income = include_no_income
    def plot_base_statistic(self,max_point=1000):
        """
        Plot total statistical figures of the sample.
        
        Note: No income sample points are excluded from 'Income distribution' and
        'Sample representative plot'.

        Parameters
        ----------
        max_point : int, optional
            Number of random data points to plot in scatter plot.

        """
        
        df = self.df
        
        gs0 = gridspec.GridSpec(3,3, height_ratios=[1,1,1.5])
        fig = plt.figure(figsize=(14,12))
        fig.suptitle("General statistics",fontsize=20,fontweight='bold')
        ax1 = fig.add_subplot(gs0[0,0],box_aspect=0.8,title='Income distribution')
        ax2 = fig.add_subplot(gs0[0,1],box_aspect=0.8,title='Age distribution')
        ax3 = fig.add_subplot(gs0[0,2],box_aspect=0.8,title='Education level distribution')
        ax4 = fig.add_subplot(gs0[1,0],box_aspect=0.8,title='Gender ratio')
        if self._include_no_income:
            ax5 = fig.add_subplot(gs0[1,1],box_aspect=0.8,title='Working status')
        ax6 = fig.add_subplot(gs0[2,:],title='Sample representative plot')
        
        #plot only data point with nonzero income
        ax1.hist(df.income.loc[df.income>0],bins=40)
        ax2.hist(df.age,bins=40)
        ax3.hist(df.education,bins=40)
        
        gender_df = self.df.groupby('gender').count()
        gender_count = gender_df['income'].to_numpy()
        gender_labels = gender_df.index.to_numpy()
        ax4.pie(gender_count,labels=gender_labels,autopct="%0.2f%%")
        
        if self._include_no_income:
            work_stt_df = self.df.groupby('work_status').count()
            work_count = work_stt_df['income'].to_numpy()
            work_labels = work_stt_df.index.to_numpy()
            work_labels[work_labels=='Yes']='In working age'
            work_labels[work_labels=='No']='Not in working age'
            ax5.pie(work_count,labels=work_labels,autopct="%0.2f%%")
        buffer_array = df.loc[df.income>0].to_numpy()
        #set number of plot points
        n_buffer = buffer_array.shape[0]
        if max_point>n_buffer:
            n_scatter = buffer_array
        else:
            #get random stratify from sample    
            n_scatter, _ = train_test_split(buffer_array,
                                            train_size=max_point,
                                            stratify=buffer_array[:,2])
        
        #plot only 'max_point' points by default
        for item in df.education.unique():
            ax6.scatter(*n_scatter[n_scatter[:,2]==item][:,[1,0]].T,
                        label=f'{item}')
        ax6.legend()
        plt.tight_layout()
        
    def plot_partial_statistic(self,max_point=500):
        """
        Plot statistical figures of each education level

        Parameters
        ----------
        max_point : int, optional
            Number of random data points to plot in scatter plot.

        """
        #exclude no income from plot
        df = self.df.loc[self.df.income>0]
        
        edu_lvl = df.education.unique()
        n = df.education.unique().size
        fig = plt.figure(constrained_layout=True,
                         figsize=(15,15))
        row_subfigs = fig.subfigures(n, 1) 
        
        for i,(subfig,item) in enumerate(zip(row_subfigs,edu_lvl)):
            buffer_df = df.loc[df.education==item]
            subfig.suptitle(f"Education level: {item}",
                            fontsize=20,fontweight='bold')
            ax = subfig.subplots(1, 4) 
            
            ax[0].hist(buffer_df.income,bins=40)
            ax[1].hist(buffer_df.age,bins=40)
            
            buffer_array = buffer_df.to_numpy()
            
            #set number of plot points
            n_buffer = buffer_array.shape[0]
            if max_point>n_buffer:
                n_scatter = buffer_array
            else:
                #get random stratify from sample
                n_scatter, _ = train_test_split(buffer_array,
                                                train_size=max_point,
                                                stratify=buffer_array[:,2])
            
            ax[2].scatter(*n_scatter[:,[1,0]].T)
            ax[2].set_xlabel("Age")
            ax[2].set_ylabel("Income")
            
            gender_df = buffer_df.groupby('gender').count()
            gender_count = gender_df['income'].to_numpy()
            gender_labels = gender_df.index.to_numpy()
            ax[3].pie(gender_count,labels=gender_labels,autopct="%0.2f%%")
            if i==0:
                ax[0].title.set_text("Income distribution")
                ax[1].title.set_text("Age distribution")
                ax[2].title.set_text("Sample representative plot")
                ax[3].title.set_text("Gender ratio")
        fig.show()
        
    def plot_input_distribution(self,visual_points=10000,bins=60):
        """
        Visualize the input data based on input type.

        Parameters
        ----------
        visual_points : int, optional
            Number of plot points used to create visualization if input type
            is 'normalize'.
        bins : int, optional
            Number of bins for smoothness

        """
        fig,ax = plt.subplots(3,1,figsize=(8,16))
        data_list = [self._income_data,self._age_data]
        for i,(data,data_name) in enumerate(zip(data_list,['Income','Age'])):
            if data[0]=='ratio':
                label = np.array(data[1])
                height = np.array(data[2])
                height = height/height.sum()
                ax[i].bar(x=label,height=height)
            elif data=='array':
                ax[i].hist(data[1],bins=bins)
            elif data[0]=='normalize':
                ax[i].hist(np.random.normal(data[1],
                                            data[2],
                                            size=visual_points),
                           bins=bins)
            ax[i].title.set_text(f"{data_name} distribution")
        #education level plot
        label = list(self.education_lvl.keys())
        height = np.array([item[0] for item in self.education_lvl.values()])
        height = height/height.sum()
        ax[-1].bar(x=label,height=height)
        ax[-1].title.set_text("Education distribution")
        fig.show()
    def sample_statistics(self,include_edu_corr=True,include_gender_corr=False):
        """
        Print the statistical data of the sample, data of no income population 
        are excluded from the correlation calcuation if any. 
        
        Parameters
        ----------
        include_edu_corr : bool, optional
            Keep education column as numerical data (0,1,2,3,4,....) to allow 
            caculation of correlation between education level, income, age in
            total database. Higher number mean higher level of education.
        include_gender_corr: bool, optional
            Keep gender column as numerical data (0,1,2,3,4,....) to allow 
            caculation of correlation. Normally this can be ignored.
        """
        df = self.df.copy()
        #exclude no income data points in statistics
        df = df.loc[df.income>0]
        if include_gender_corr:
            df.gender.replace({"male":0,"female":1},inplace=True)
        if not include_edu_corr:
            stat3 = df.groupby('education').corr()
            stat4 = df.groupby('education').describe().T
            stat1 = df.corr()
            stat2 = df.describe()
        else:
            stat3 = df.groupby('education').corr()
            stat4 = df.groupby('education').describe().T
            numeric_edu = range(df.education.unique().size)
            df.education.replace(dict(zip(self.education_lvl.keys(),
                                          numeric_edu)),
                                 inplace=True)
            stat1 = df.corr()
            stat2 = df.describe()
            
        print(f"""
Main data correlation:
--------------------------------------------------------
{stat1}
--------------------------------------------------------
Main data descriptive statistics
--------------------------------------------------------
{stat2}
--------------------------------------------------------


Partial data correlation by education level
--------------------------------------------------------
{stat3}
--------------------------------------------------------
Partial data descriptive statistics by education level
--------------------------------------------------------
{stat4}
--------------------------------------------------------
              """)
    
def convert_covar_to_corr(covar_matrix):
    """
    Convert the covariance matrix to correlation matrix.

    Parameters
    ----------
    covar_matrix : np.ndarray
        Covariance matrix.

    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix.

    """
    if covar_matrix.ndim!=2:
        raise ValueError("'covar_matrix' only take 2-D array")
    if covar_matrix.shape[0]!=covar_matrix.shape[1]:
        raise ValueError("'covar_matrix' only take (n x n) shape array")
    
    std_diag = np.diag(np.sqrt(covar_matrix.diagonal()))
    invert_std = np.linalg.inv(std_diag)
    corr_matrix = invert_std.dot(covar_matrix).dot(invert_std)
    return corr_matrix

def convert_corr_to_covar(corr_matrix,std_diag):
    """
    Convert the correlation matrix to covariance matrix.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix.
    std_diag : np.ndarray
        Diagonal matrix of each variable standard deviations.

    Returns
    -------
    covar_matrix : np.ndarray
        Covariance matrix.

    """
    if corr_matrix.ndim!=2:
        raise ValueError("'corr_matrix' only take 2-D array.")
    if corr_matrix.shape[0]!=corr_matrix.shape[1]:
        raise ValueError("'corr_matrix' only take (n x n) shape array.")
    
    if std_diag.ndim!=2:
        raise ValueError("'std_diag' only take 2-D array.")
    if std_diag.shape[0]!=std_diag.shape[1]:
        raise ValueError("'std_diag' only take (n x n) shape array.")
    if np.count_nonzero(std_diag)-std_diag.diagonal().size>0:
        raise ValueError("'std_diag' must be  diagonal array.")
    
    covar_matrix = std_diag.dot(corr_matrix).dot(std_diag)
    return covar_matrix

def create_corr_matrix(required_corr,remain_corr=0.0,seed=None):
    """
    Create the correlation matrix of n variables with the first variable is the 
    target variable, which means its correlation with other variables is 
    required to input manually, all correlations between the remaining 
    variables can be either set manually or randomly.

    Parameters
    ----------
    required_corr : list|1-D np.ndarray
        Required correlations between target variable and all other variables.
    remain_corr : str|list|1-D np.ndarray|float, optional
        If remain_corr = "random", the correlations between all remaining 
        variables will be randomly set between -1 and 1.
        
        If remain_corr = float number, all correlations between remaining 
        variables will be set to that value.
        
        If remain_corr = list|np.ndarray, the correlation between remaining
        variables will be set according to the list|array as follow:
            [corr(x2,x3),corr(x2,x4),..,corr(x2,xn),
             corr(x3,x4),..corr(xn-1),corr(xn)]
    seed : int, optional
        Seed for random generation if 'remain_corr'='random'.

    Returns
    -------
    Correlation matrix: np.ndarray

    """
    check_arr = np.array(required_corr,dtype=np.float64)
    if np.any(check_arr>1) or np.any(check_arr<-1):
        raise ValueError("'required_corr' correlations must be between [-1,1].")
    
    if type(remain_corr)==str:
        if remain_corr!="random":
            raise ValueError("Input method isn't supported for 'remain_corr'.")
    elif type(remain_corr) in [float,int]:
        if remain_corr>1 or remain_corr<-1:
            raise ValueError("'remain_corr' input value must be between [-1,1].")
    else:
        remain_check_arr = np.array(remain_corr,dtype=np.float64)
        if np.any(remain_check_arr>1) or np.any(remain_check_arr<-1):
            raise ValueError("All 'remain_corr' correlations must be between [-1,1].")
        if remain_check_arr.size != np.arange(1,check_arr.size).sum():
            raise ValueError("Number of correlation values in 'remain_corr'"+
                             " array doesn't meet requirement")
    #No. of variables
    n = len(required_corr) + 1
    
    #diagonal matrix
    buffer_array = np.diag(np.full(n,1)).astype(np.float64)
    
    buffer_array[1:,0] = required_corr
    
    if n-1>1: #if nmore than 2 variable
        if type(remain_corr)==str:
            np.random.seed(seed)
            extra_corr = np.random.uniform(-1,1,np.arange(1,n-1).sum())
        elif type(remain_corr) in [float,int]:
            extra_corr = np.full(np.arange(1,n-1).sum(),remain_corr)
        else:
            extra_corr = np.array(remain_corr,dtype=np.float64)
    
        triu_idx = np.triu_indices(n,k=1)
        
        buffer_array[(triu_idx[0][n-1:],triu_idx[1][n-1:])]=extra_corr
    
    buffer_array = buffer_array + buffer_array.T - np.diag(np.diag(buffer_array))
    
    return buffer_array

def age_distribution(left=0,mode=30,std=10,right=80,
                     sample=10000000,norm_dist_weight=0.5,
                     seed=None):
    """
    Simulating a population age distribution using normal distribution combined
    with triangular distribution.

    Parameters
    ----------
    left : int, optional
        Lower limit for triangular distribution.
    mode : int, optional
        Mode value of the model.
    std : int, optional
        Standard deviation for normal distribution.
    right : int, optional
        Upper limit for triangular distribution.
    sample : int, optional
        Number of sample for distribution.
    norm_dist_weight : float, optional
        The percent of normal distribution samples from 'sample'.
    seed : int, optional
        Seed for random generation.

    Returns
    -------
    dict
        Return a dictionary of {age_distribution,
                                partial_normal_distribution,
                                partial_triangualr_distribution}

    """
    np.random.seed(seed)
    dist1 = np.random.normal(mode,std,int(sample*norm_dist_weight))
    dist1 = dist1[dist1>=0]
    dist2 = np.random.triangular(left=left,mode=mode,right=right,
                                 size=int(sample*(1-norm_dist_weight)))
    
    return {"age_distribution":np.concatenate((dist1,dist2)),
            "partial_normal_distribution":dist1,
            "partial_triangualr_distribution":dist2}

def combined_distribution(N,properties,sample_ratio,seed=None):
    """
    Combine multiple numpy.random |scipy.stats distribution with different 
    sample size to get a customized distribution.

    Parameters
    ----------
    N : int
        Total sample size.
    properties : dict
        Dictionary of distributions used to combine. Note that 'size' parameter
        mustn't be included inside the item inputs.
        Syntax: {'distribution_1_name':[*arg1,**kwargs1],
                 'distribution_2_name':[*arg2,**kwargs2]}
        E.g: {'np.random.normal':[(),{'loc':5,'scale':3}],
              'scipy.skewnorm.rvs':[(-1,50,30),{}]}
    sample_ratio : list of int|float
        List of distribution ratios. E.g: [0.3,0.2,0.5]
    seed : int, optional
        Seed for random generation.

    Returns
    -------
    return_dict : dict
        Return the combined distribution and all component distributions.

    """
    
    if len(properties)!=len(sample_ratio):
        raise ValueError("'properties' and 'sample_ratio' must have match length.")
    
    np.random.seed(seed)
    sample_ratio = np.array(sample_ratio)/np.array(sample_ratio).sum()
    
    container = []
    return_dict = {'combined_distibution':None}
    
    for i,item in enumerate(properties.items()):
        func = eval(item[0])
        item[1][1]['size']=round(N*sample_ratio[i])
        dist = func(*item[1][0],**item[1][1])
        container.append(dist)
        return_dict[item[0]]=dist
        
    return_dict['combined_distibution']=np.concatenate(container)
    
    return return_dict

class sampling_pipeline:
    def __init__(self,df,sampling):
        pass

#How many time a consumer will consume a FMCG product over a set of time
class consumption_sample: 
    def __init__(self,input_df,
                 mean_unit,
                 std_unit,
                 income_factor=0.5,
                 age_factor=0.5,
                 extra_factors=None,
                 constraint=None,
                 outlier_p=None):
        self._input_df = input_df
        
        #income factor
        #product_price
        
        
        
        