#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:10:04 2023

@author: ab
"""

# This module implements analytical approximations to the normal CDF to speed-up the originaly slow scipy.stats.norm.cdf() function.
# See this article and table inside: DOI:10.3934/math.2022648
# To use approximations always call the 'manager' function 'proxNormCDF()''. This function, in turn, calls specific approximations and handles negative arguments.
# (note, approximation expressions are defined for positive arguments only; handling of negative arguments is managed by the in the calling function proxNormCDF())
# You can change the approximation expression by correspondingly editing the name of the function called inside the ''manager''
#
# THIS is a SCALAR VERSION !!!  Implementation foresees that parameters x, mu, and sigma should be scalars
# Vector version is also available but in hazard curve calculation tests appeared to be significantly slower


import numpy as np

# This is manager function. Always call approximations via this function. Change approximation model by editing the name of the model inside.
# For approximation models see table in DOI:10.3934/math.2022648
def proxNormCDF( x, mu, sigma ):
  z = (x-mu)/sigma
  if( z > 0 ):
    fi = proxNormCDF_linear(z)
  elif( z < 0 ):
    fi = 1. - proxNormCDF_linear(-z)
  else:
    fi = 0.5
  return fi


# Simple linear interpolation between tabulated CDF values
# xMin = 0.; xMax = 3.; number of steps = 15
# Maximum approximation error: 1.3e-3
# Define some constants as global to compute them only once
xStep = 0.2
yTab = [ 0.5, 0.57925971, 0.65542174, 0.72574688, 0.7881446, 0.84134475,
								 0.88493033, 0.91924334, 0.94520071, 0.96406968, 0.97724987,
								 0.98609655, 0.99180246, 0.99533881, 0.99744487, 0.9986501 ]
nSteps = len(yTab)-1
def proxNormCDF_linear( x ):
  # Note that the expression defined for positive argument only
  global nSteps,xStep,yTab
  idx  = int(x/xStep)
  if( idx >= nSteps ):
    fi = 1. #yTab[-1]
  else:
    fi = yTab[idx] + (yTab[idx+1]-yTab[idx])*(x/xStep-idx)
  return fi


### Below are various CDF approximation expressions as given in the table in DOI:10.3934/math.2022648
# Numbers refer to the reference numbers in the above mentioned publication
# Also note that approximation expressions are defined for positive argument vaues only. Handling of negative arguments is managed by 'manager' proxNormCDF()

def proxNormCDF_14( x ):
  # Derenzo, 1977: max error 7e-5
  # Note that the expression defined for positive argument only
  return ( 1. - 0.5*np.exp( -(x*(83*x+351)+562)/(703/x+165) ) )

def proxNormCDF_17( x ):
  # Lin, 1989: max error 6e-3
  return ( 1. - 0.5*np.exp(-0.717*x-0.416*x*x) )

def proxNormCDF_25a( x ):
  # Bowling, 2009a: max error 1e-2
  return ( 1./(1.+np.exp(-1.702*x)) )

def proxNormCDF_25b( x ):
  # Bowling, 2009b: max error 1e-4
  return ( 1./(1.+np.exp(-0.07056*x*x*x-1.5976*x)) )

def proxNormCDF_28( x ):
  # Eidous and Al-Saman, 2016: max error 2e-3
  return ( 0.5 + 0.5*np.sqrt(1.-np.exp(-5./8.*x*x)) )

