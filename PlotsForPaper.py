## Specifically for final paper
# Author: Shion Andrew

import math
import statistics
import csv
import os

# MatPlotlib
from matplotlib import pylab
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib import rc

# Scientific libraries
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

# Astropy and Gaia
import astroquery
import keyring
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy import stats

def plotLightCurve():
    magerr = []
    mag = []
    abjd = []
    '''
    #Sample Variabe 1 at  (43.41934044825248,7.650638497912617), ID 8382715305512064
    filename = 'SampleVariable1.csv'
    P = 1.1165359
    t0 = 54062.26802

    #Sample Variable 2 at (37.959007075584914,9.538550416005373)
    filename = 'SampleVariable2.csv'
    P = 0.4664226
    t0 = 53725.17696
    
    #Sample Variable 3 at (43.69560029174401, 16.44192897023445)
    filename = 'SampleVariable3.csv'
    P = 1.0638116 
    t0 = 53728.19668

     
    #Sample Variable 4 at (58.348628263617, 14.665068354756785)
    filename = 'SampleVariable4.csv'
    P = 0.6385608
    t0 = 53728.23433
    
    #Sample Variable 5 at (58.348628263617,14.665068354756785)
    filename = 'SampleVariable5.csv'
    P = 0.3851824
    t0 = 53728.23326   
    
    #Sample Variable 6 at (58.128312500465285, 17.590598901970537)
    filename = 'SampleVariable6.csv'
    P = 0.4643610
    t0 = 53728.23326 
    
    #Sample Variable 7 at (63.90131988835955, 21.97055897685485)
    filename = 'SampleVariable7.csv'
    P = 0.4298898
    t0 = 53469.12043
    
    #Sample Variable 8 at (61.04671835294998, 22.599476443278917)
    filename = 'SampleVariable8.csv'
    P = 0.3675693 
    t0 = 53732.17356
    
    #Sample Variable 9 at (49.690289026300846, 21.576856893960613)
    filename = 'SampleVariable9.csv'
    #P = 0.8867552
    P = 1.7735104 
    t0 = 53728.19932
    
    #Sample Variable 10 at (57.005321265437814,23.827051423646)
    filename = 'SampleVariable10.csv'
    P = 1.2420231
    t0 = 53702.31836
    
    #Sample Variable 11 at (60.29559468726097, 23.888750503691387)
    filename = 'SampleVariable11.csv'
    P = 0.6201218
    t0 = 54093.21929
    
    #Sample Variable 12 at (26.520695898243762	, 19.115719804180774)
    filename = 'SampleVariable12.csv'
    P = 1.4751568
    t0 = 53706.29390 
    
    #Sample Variable 13 at (44.59795304214619,28.579830947900444)
    filename = 'SampleVariable13.csv'
    P =  0.9289320
    t0 = 53707.21382
    
    #Sample Variable 14 at (47.38848869191275, 31.04802726565917)
    filename = 'SampleVariable14.csv'
    P = 1.7227780
    t0 = 54041.32134 
    
    #Sample Variable 15 at (38.72674646352375,33.25185966486731)
    filename = 'SampleVariable15.csv'
    P = 1.5985178
    t0 = 53707.19538
    
    #Sample Variable 16 at (
    filename = 'SampleVariable16.csv'
    P = 0.5642188
    t0 = 53708.25039 

    #Sample Variable 17 at (70.03337935576555,26.090308893832937)
    filename = 'SampleVariable17.csv'
    P = 1.8179476
    t0 = 53470.12226

    #Sample Variable 18 at (65.5587174075488,26.728914141050165) 
    filename = 'SampleVariable18.csv'
    P = 1.1222842	
    t0 = 53734.24409'''
    
    #Sample Variable 19 at (65.9365430976055,26.861719404764827)
    filename = 'SampleVariable19.csv'
    P = 3.1293268 
    t0 = 53470.12172


    with open('/Users/touatokuchi/Desktop/MSU/PlotsForPaper/' + filename) as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            try:
                mag.append(float(row[1]))
                magerr.append(float(row[2]))
                abjd.append(float(row[5]))
            except:
                print("error")

    #False Positive at 44.90148, 0.66384
    #P = 0.4212433
    #t0 = 53710.24116
    #P = 1.0750575
    #P = 1.2996582
    #t0 = 53611.39860

    aphase = []
    for i in range(len(abjd)):
        aphase.append(foldAt(abjd[i],P,T0=t0,getEpoch=False))

    mag2 = []
    aphase2 = []
    magerr2 = []
    for i in range(len(aphase)):
        mag2.append(mag[i])
        magerr2.append(magerr[i])
        aphase2.append(aphase[i]+1)
    mag = mag + mag2
    aphase = aphase + aphase2
    magerr = magerr + magerr2

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylabel(r'\textbf{Magnitude}')
    plt.xlabel(r'\textbf{Phase}')
    plt.gca().invert_yaxis()
    plt.ylim(21.5, 13)
    plt.scatter(aphase, mag, color = 'red', s = .5)
    plt.errorbar(aphase, mag, yerr=magerr, ecolor = 'black', lw = .6, capsize = 1, fmt = 'none')
    plt.show()

def foldAt(time, period, T0=0.0, getEpoch=False):
    epoch = np.floor( (time - T0)/period )
    phase = (time - T0)/period - epoch
    if getEpoch:
        return phase, epoch
    return phase

def main():
    plotLightCurve()

if __name__== "__main__":
    main()
