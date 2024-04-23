#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:22:28 2024

@author: Giorgio
"""


import os
#import sys

#print(sys.getenv)
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/'

os.environ['PATH'] += ':/Library/Frameworks/R.framework/Versions/4.3-arm64/Resources/bin'


import rpy2
print(rpy2.__version__)

import rpy2.situation
for row in rpy2.situation.iter_info():
    print(row)

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

# import R's "utils" package
utils = importr('utils')


# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)

# R package names
packnames = ('ggplot2', 'hexbin')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
   
    
import rpy2.robjects as robjects
pi = robjects.r['pi']
print(pi[0])

robjects.r('''
        # create a function `f`
        f <- function(r, verbose=TRUE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            return(2 * pi * r)
        }
        # call the function `f` with argument value 3
        f(3)
        ''')
        
r_f = robjects.globalenv['f']        
print(r_f.r_repr())      
res = r_f(3)       
print(res)      
        
        
rsum = robjects.r['sum']
gino = rsum(robjects.IntVector([10,11,20]))
print(gino)      
        
import rpy2.robjects as robjects

r = robjects.r

x = robjects.IntVector(range(10))
y = r.rnorm(10)

r.X11()

r.layout(r.matrix(robjects.IntVector([1,2,3,2]), nrow=2, ncol=2))
r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")

# import rpy2.robjects as robjects
# from rpy2.robjects import Formula, Environment
# from rpy2.robjects.vectors import IntVector, FloatVector
# from rpy2.robjects.lib import grid
# from rpy2.robjects.packages import importr, data
# from rpy2.rinterface_lib.embedded import RRuntimeError
# import warnings

# # The R 'print' function
# rprint = robjects.globalenv.find("print")
# stats = importr('stats')
# grdevices = importr('grDevices')
# base = importr('base')
# datasets = importr('datasets')
       
# import math, datetime
# import rpy2.robjects.lib.ggplot2 as ggplot2
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# base = importr('base')

# mtcars = data(datasets).fetch('mtcars')['mtcars']        
# pp = (ggplot2.ggplot(mtcars) +
#       ggplot2.aes_string(x='wt', y='mpg', col='factor(cyl)') +
#       ggplot2.geom_point() +
#       ggplot2.geom_smooth(ggplot2.aes_string(group = 'cyl'),
#                           method = 'lm'))
# pp.plot()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        