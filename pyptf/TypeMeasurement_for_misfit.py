"""
First approaches for a tsunami mismatch evaluator for the WP6 workflow. Based on the GITEWS approach.
M. Bänsch (08/2023)

This file contains the definitions for the classes
"""

import numpy as np

class TypeMeasurement:
  """
  Object to classify the types of measurement 
  
  Class members:
    nstations                   : number of measurement stations
    stationcoordinates    : coordinates for each measurement station
    stationweights          : weights for each station (gives a measure of confidence for the data)
    distancerange          : range (min, max) of measured distance of the data
    scaling                      : scaling factor to normalize the data (see scale_measured_data)
    measuretypeweight  : weighting factor for the respective measure type (e.g. GNSS)
  """

  def __init__(self, nstations, stationcoordinates, stationweights, distancerange, scaling=1, measuretypeweight=1):
    self.nstations = nstations
    self.stationcoordinates = stationcoordinates
    self.stationweights = stationweights
    self.distancerange = distancerange
    self.scaling = scaling
    self.measuretypeweight = measuretypeweight
  
  
  def scale_measured_data(self, distmax=1):
    """
    Function to calculate the scaling factor for the data according to the approach as in GITEWS.
      The data is mapped onto the unit interval [0,1] in such a way that the first 25% 
      of the measured distance are mapped onto [0, 0.8]. 
    """   
    self.scaling = 4. / distmax * np.tan(0.8 * np.pi / 2.)
    
    return
  
  
  def scaling_func(self, dist):
    """
    Function to calculate the scaling according to the approach as in GITEWS
      which is given by 
          scaling_func = arctan(scaling * distance) * 2 / pi
      Needs the scaling factor to be calculated prior to calling
    """
    
    return np.arctan(self.scaling * dist) * 2. / np.pi
  
  
  def renormalize_stationweights(self):
    """
    Function to renormalize the weights for each station if one (or more) have to be set to zero
    """
    grp_weight = sum(self.stationweights)
    self.stationweights /= grp_weight
      
    return
  
  
  def renormalize_typeweight(self, typeweightsum):
    """
    Function to renormalize the weight for this type of measurement (e.g. GNSS)
    """
    self.measuretypeweight /= typeweightsum
      
    return    
