import numpy as np
cimport numpy as np
cimport cython
np.import_array()
ctypedef np.float64_t float64_t
ctypedef np.int64_t int64_t

#------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def arrange_value_to_array(np.ndarray[int64_t,ndim=1] input_value,
                           unsigned int array_len,
                           np.ndarray[float64_t,ndim=1] init_value=None,
                           np.ndarray[float64_t,ndim=2] init_array=None,
                           np.ndarray p=None,
                           np.ndarray upper_lim=None,
                           unsigned int method_limit=30,
                           seed=None):
    """
    arrange_value_to_array(input_value,
                           array_len,
                           init_value=None,
                           init_array=None,
                           p=None,
                           upper_lim=None,
                           method_limit=30,
                           seed=None):
    
    Fill whole number to an array (like putting beans into a set of glass). 
    Fill only up to the limit of each position. This function designed to work 
    only on 1-D|2-D array.

    Parameters
    ----------
    input_value : 1-D np.ndarray
        Number value to fill in the array, can only have size=1 if 'init_value'
        is used. Each array element is equal to the amount use to fill for an 
        array row if 'init_array' is used.
        
    array_len : integer
        Length of fill-in array.
        
    init_value : 1-D float np.ndarray, optional
        Default numbers inside the 1-D array. Decimal will be ignored.
        If None, initial values is 0, also don't take infinite value.
        
    init_array: 2-D float np.ndarray, optional
        Default numbers inside the 2-D array. Decimal will be ignored.
        If None, initial values is 0, also don't take infinite value.
        Each element along the first axis will be used for fill in.
        
    p : 1-D|2-D float np.ndarray, optional
        Distribution ratio, can be either percentage or not.
        1-D array will be broadcast if 'init_array' is used.
        2-D array input must match 'init_array' shape.
        If None, distribution is even for all cases.
        NOTE: Calculation won't be based on distribution probability but 
        expectation (E=sigma(p*x)).Therefore, with small sample, result might 
        not as expected, but calculation for big sample is faster than using 
        numpy.random.choice.
        
    upper_lim : 1-D|2-D float np.ndarray, optional
        Upper limit of each position. Decimal will be ignored.
        1-D array will be broadcast if 'init_array' is used.
        2-D array input must match 'init_array' shape.
        If None, default limit is infinite.
        
    method_limit: int, optional
        The up to limit which np.random.choice will be used instead of fast 
        filling method. The main purpose of this is to keep the randomness to a
        certain degree if necessary but still keep a fast calculation speed.
    
    seed: int, optional
        Seed for random generation. If 'method_limit'=0, seed won't be applied.
        
    Returns
    -------
        (loop_array,remain_array)
    
    loop_array : 2-D array
        Return array with fill-in values.
        
    remain_value : 1-D np.ndarray
        The remain values not filled.
    """
    cdef np.ndarray initial_array
    cdef np.ndarray[float64_t,ndim=1] new_p,fractional,integral,trim_array,remain_array
    cdef np.ndarray[float64_t,ndim=2] loop_array,loop_p,loop_up_lim
    cdef np.ndarray[int64_t,ndim=1] match_idx,rank,random_choice,counts
    cdef int i,n,loop_length
    cdef float current_sum,lim_sum,remain_value
    cdef object rng_gen
    
    #condition for input between {input_value,init_value,init_array}
    if input_value.shape[0] == 0:
        raise ValueError("'input_value' can't be empty.")
    elif np.any(input_value<0):
        raise ValueError("'input_value' don't take negative values.")
    elif not np.all(np.isfinite(input_value)):
        raise ValueError("'input_value' don't take infinite values.") 
    
    if (init_value is not None) and (init_array is not None):
        raise TypeError("'init_value' and 'init_array' can't be used at the "\
                        "same time.")
    elif (init_value is None) and (init_array is None): 
        if input_value.shape[0] == 1:
            initial_array = np.zeros(array_len)
        else:
            initial_array = np.zeros((input_value.shape[0],array_len))
    elif init_value is not None: #use for scalar init
        if input_value.shape[0] == 1:
            #dont take infinity initial values.
            initial_array = np.floor(init_value)
            if initial_array.shape[0] != array_len:
                raise ValueError("'init_value' length should match 'array_len'.")
            if not np.all(np.isfinite(initial_array)):
                raise ValueError("'init_value' don't take infinite values.")
        else:
            raise ValueError("'input_value' length must be 1 when using "\
                             "'init_value'.")
    else: #use for 2-D array init input
        if input_value.shape[0] == init_array.shape[0]:
            #dont take infinity initial values.
            initial_array = np.floor(init_array)
            if initial_array.shape[1] != array_len:
                raise ValueError("'init_array' 2-D length should match "\
                                 "'array_len'.")
            if not np.all(np.isfinite(initial_array)):
                raise ValueError("'init_array' don't take infinite values.")
        else:
            raise ValueError("'input_value' array length should match "\
                             "'initial_array' 1nd dimension length.")
    #initial condition for input p
    if p is not None:
        if p.dtype.kind != 'f':
            raise TypeError("'p' array only take np.float64 values.")
        elif p.ndim not in [1,2]:
            raise ValueError("'p' array only take 1 or 2-D array.")
    #set condition for compatitble with init_value and init_array
    if initial_array.ndim ==1:
        if p is None:
            p=np.full(array_len,1.0)
        elif p.ndim != 1:
            raise ValueError("'p' dimension doesn't match 'init_value'.")
        elif p.shape[0]!=array_len:
            raise ValueError("'p' length doesn't match 'array_len'.")
    else:
        if p is None:
            p = np.full((initial_array.shape[0],array_len),1.0)
        elif (p.ndim == 2) and (p.shape[1]!=array_len):
            raise ValueError("'p' 2nd-Dimension length doesn't match 'array_len'.")
        elif (p.ndim == 2) and (p.shape[0]!=input_value.shape[0]):
            raise ValueError("'p' 1nd-Dimension length doesn't match "\
                             "'input_value' length.")
        elif (p.ndim == 1) and p.shape[0]!=array_len:
            raise ValueError("'p' length doesn't match 'array_len'.")
    #initial condition for input upper_lim
    if upper_lim is not None:
        if upper_lim.dtype.kind != 'f':
            raise TypeError("'upper_lim' array only take np.float64 values.")
        elif upper_lim.ndim not in [1,2]:
            raise ValueError("'upper_lim' array only take 1 or 2-D array.")
    
    #set condition for compatitble with init_value and init_array
    if initial_array.ndim ==1:
        if upper_lim is None:
            upper_lim = np.full(array_len,np.inf)
        elif upper_lim.ndim != 1:
            raise ValueError("'upper_lim' dimension doesn't match 'init_value'.")
        elif upper_lim.shape[0]!=array_len:
            raise ValueError("'upper_lim' length doesn't match 'array_len'.")
        upper_lim=np.floor(upper_lim)
    else:
        if upper_lim is None:
            upper_lim = np.full((initial_array.shape[0],array_len),np.inf)
        elif (upper_lim.ndim == 2) and (upper_lim.shape[1]!=array_len):
            raise ValueError("'upper_lim' 2nd-Dimension length doesn't match 'array_len'.")
        elif (upper_lim.ndim == 2) and (upper_lim.shape[0]!=input_value.shape[0]):
            raise ValueError("'upper_lim' 1nd-Dimension length doesn't match "\
                             "'input_value' length.")
        elif (upper_lim.ndim == 1) and upper_lim.shape[0]!=array_len:
            raise ValueError("'upper_lim' length doesn't match 'array_len'.")
        upper_lim=np.floor(upper_lim)
                    
    if np.any(initial_array>upper_lim):
        raise ValueError("init values must be smaller|equal comparing to limits.")
    
    #sychronize calculation when using either 'init_value' or 'init_array'
    if input_value.shape[0]==1 and ((init_value is not None) or \
                                (init_value is None) and (init_array is None)):
        loop_length = 1
        loop_array = initial_array[np.newaxis,:]
        loop_p = p[np.newaxis,:]
        loop_up_lim = upper_lim[np.newaxis,:]
    else:
        loop_length = input_value.shape[0]
        loop_array = initial_array
        if p.ndim ==1:
            loop_p = np.full((input_value.shape[0],array_len),p)
        else:
            loop_p = p
        if upper_lim.ndim ==1:
            loop_up_lim = np.full((input_value.shape[0],array_len),upper_lim)
        else:
            loop_up_lim = upper_lim
    
    
    remain_array = np.array([])
    for i in range(loop_length):
        #maximum fill in amount for each loop
        lim_sum = np.where(loop_p[i]>0,loop_up_lim[i],0).sum()
        remain_value = float(input_value[i]) #remain value to fill after each while loop
        current_sum = loop_array[i].sum() #array current sum
    
        if input_value[i] >= method_limit:
            #only stop when no more input value or array is full.
            while remain_value>0 and current_sum<lim_sum: 
                #only add to avalable idx with upper_lim available and p>0    
                match_idx = np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))[0]
                new_p = loop_p[i][np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))]
                new_p = (new_p/new_p.sum())
                
                #separate the fractionl and integral part
                fractional,integral = np.modf(new_p*remain_value)
                loop_array[i,match_idx]+=integral
                
                n = fractional.sum().round().astype(np.int64)
                
                #add rounded fractional part to index with highest fractional without 
                #exceed total input_value
                if n>0: 
                    rank = fractional.argsort()
                    loop_array[i,match_idx[rank][-n:]] +=1
                
                #clip any excess value compare to upper_lim
                trim_array=np.clip(loop_array[i],np.zeros(array_len),loop_up_lim[i]) 
                remain_value = loop_array[i].sum()-trim_array.sum()
                loop_array[i]=trim_array
                current_sum = loop_array[i].sum()
        else: # in case of need for random output to a certain degree
            rng_gen = np.random.default_rng(seed)
            while remain_value>0 and current_sum<lim_sum:
                match_idx = np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))[0]
                new_p = loop_p[i][np.where((loop_array[i]<loop_up_lim[i]) &(loop_p[i]>0))]
                new_p = (new_p/new_p.sum())
                random_choice = rng_gen.choice(match_idx,int(remain_value),p=new_p)
                match_idx,counts = np.unique(random_choice,return_counts=True)
                
                loop_array[i,match_idx] += counts
                    
                trim_array=np.clip(loop_array[i],np.zeros(array_len),loop_up_lim[i]) 
                remain_value = loop_array[i].sum()-trim_array.sum()
                loop_array[i]=trim_array
                current_sum = loop_array[i].sum()
        
        remain_array = np.append(remain_array,remain_value)
    return loop_array,remain_array