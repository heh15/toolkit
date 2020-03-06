import numpy as np
import os
import scipy.ndimage as sni
import sys
import re


imagename|awk 'BEGIN(FS="\")'{for(i=1;i<=NF;i++)}if($i~/NGC/)print $i}')'

imagename_tmp=re.split('\_|\.',imagename)
imagename=imagenames[0]


m=re.search('ima.*',image_test)

imagename_tmp=re.split('\_|\.',imagename)
CO=[string for string in imagename_tmp\
    if re.match('.*CO.*',string)][0]
spw=[string for string in imagename_tmp\
     if re.match('spw.*',string)][0]


