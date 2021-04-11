def count_time(stop, start):
    '''
    Convert the time difference into human readable form. 
    '''
    dure=stop-start
    m,s=divmod(dure,60)
    h,m=divmod(m,60)
    print("%d:%02d:%02d" %(h, m, s))

    return

def initialize_value(variable, value):
    '''
    initialize the value of the variable if is not defined before. 
    '''
    try:
        variable
    except:
        variable = value

    return

from math import log10, floor
def round_sig(x, sig=2):
    '''
    round the x to certain significant figures
    '''
    return round(x, sig-int(floor(log10(abs(x))))-1)



