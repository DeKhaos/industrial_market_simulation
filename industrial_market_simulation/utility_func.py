import numpy as np
#------------------------------------------------------------------

def modifier_sequence(value,target_value,diff=0.01,n_term=10,tol=1e-5):
    """
    This function is built to be used for loops.
    
    This function will find the common ratio of a geometric sequence with a 
    starting 'value' and the final term 'check_value' to make sure 'value' is 
    as close as possible to 'target_value' after multipling a common ratio by 
    'n_term'.
    
    The purpose of this function is to slowly move 'value' closer to 'target_value'
    at an acceptable rate when needed to use 'value' as a variable for a function y=f(x) 
    comparing to final value y_final=f('target_value')
    
    Parameters
    ----------
    value : int|float
        Starting value.
    target_value : int|float
        The desired value used as the milestone for the loop sequence.
    diff : int|float, optional
        The criteria for comparing 'value' and 'target_value'.
    n_term : int, optional
        The number of term in the geometric sequence. This should decide the
        speed in which 'value' will reach 'target_value'.
    tol : float, optional
        By default 'value' can't be zero, this tolerent value is use as replacement
        instead.

    Returns
    -------
    Tuple
        (common ratio,next_value_in_sequence).
        'next_value_in_sequence' = common_ratio*value +/- diff
        Will return (None,None) if value-target_value<=diff
    """
    
    if value<0 or target_value<0:
        raise ValueError("'value' and 'target_value' must be non-negative.")
    if diff<0:
        raise ValueError("'diff' must be non-negative.")
    
    if np.abs(value-target_value)<=diff:
        return (None,None)
    
    if value==0:
        value = tol
    
    modifier = np.exp((np.log(target_value/value)/n_term))
    
    if value<target_value:
        result = value*modifier+diff
    elif value>target_value:
        result = value*modifier-diff
    else:
        result = None
        modifier = None
    return (modifier,result)

def ratio_modifier_sequence(x,x0,y_target,y=None,y0=None,
                            start_modifier = 1.15,diff=0.01,ndigits=2):
    """
    This function is built to be used for loops.
    
    The local correlation of 2 variables x and y though a transformation f 
    (f(x)=y) is taken into account to calculate the ratio between modifier=denta_x/denta_y 
    to suggest how to move x along the axis to get y closer to y_target.
    
    The purpose of this function is to slowly move x along the axis to make y 
    closer to y_target.
    
    Parameters
    ----------
    x : int|float
        x value in the current loop. Note that in the 1st loop, set x=x0 since
        x isn't calculated yet.
    x0 : int|float
        x value in the previous loop.
    y_target : int|float
        The desired y value used as the milestone for the loop sequence.
    y : int|float, optional
        y value in the current loop.
    y0 : int|float, optional
        y value in the previous loop.
    start_modifier : int/float, optional
        Set starting ratio modifier for x in the beginning, since the y0 isn't
        available in the 1st loop.
    diff : int/float, optional
        Non-negative difference allowed for variable y.
    ndigits : int, optional
        Round modifier,estimated_next_x to a given precision in decimal digits.

    Returns
    -------
    Tuple
        (modifier,estimated_next_x).

    """
    assert diff>=0,"'diff' for variable y should be >0."
    
    if x==x0:
        return round(start_modifier,ndigits),round(x0*start_modifier,ndigits)
    else:
        if y0 is None or y is None:
            return -1,None
        diff_y = y_target-y
        diff_base_y = y0-y
        diff_x = x0-x
        if np.abs(diff_y)<=diff:
            return None,None
        modifier = diff_x/diff_base_y
        return round(modifier,ndigits),round(diff_y*modifier+x,ndigits)
        