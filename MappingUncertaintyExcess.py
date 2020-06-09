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
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

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

def plotCatalinaUncertainties():

    dtype = [('Magnitude', float), ('Amplitude', float), ('SigmaOffBaseline', float)]
    finalValues = []
    with open('/Users/touatokuchi/Desktop/MSU/KnownVariablesTest/Catalina.csv') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            try:
                Vamp = float(row[17])
                magnitude = float(row[3])
                sigma = float(row[11])
                period = float(row[16])
                ## period cuts
                if period < 10:
                    finalValues.append((magnitude, Vamp, sigma))

            except ValueError:
                  continue

    csvfile.close()
    ## create array
    variableStars = np.array(finalValues, dtype=dtype)


    ampBin1 = variableStars[np.where(variableStars['Amplitude'] <= .1)]
    ampBin2 = variableStars[np.where((variableStars['Amplitude'] <= .2) & (variableStars['Amplitude'] >= .1 ))]
    ampBin3 = variableStars[np.where((variableStars['Amplitude'] <= .3) & (variableStars['Amplitude'] >= .2 ))]
    ampBin4 = variableStars[np.where((variableStars['Amplitude'] <= .4) & (variableStars['Amplitude'] >= .3 ))]
    ampBin5 = variableStars[np.where((variableStars['Amplitude'] <= .5) & (variableStars['Amplitude'] >= .4 ))]
    ampBin6 = variableStars[np.where((variableStars['Amplitude'] <= .6) & (variableStars['Amplitude'] >= .5 ))]
    ampBin7 = variableStars[np.where((variableStars['Amplitude'] <= .7) & (variableStars['Amplitude'] >= .6 ))]
    ampBin8 = variableStars[np.where((variableStars['Amplitude'] <= .8) & (variableStars['Amplitude'] >= .7 ))]
    ampBin8 = variableStars[np.where((variableStars['Amplitude'] <= .9) & (variableStars['Amplitude'] >= .8 ))]
    ampBin8 = variableStars[np.where((variableStars['Amplitude'] <= 1) & (variableStars['Amplitude'] >= .9 ))]



    #ampBin0 = variableStars[np.where(variableStars['Amplitude'] <= .05)]
    ampBin1 = variableStars[np.where((variableStars['Amplitude'] <= .1) & (variableStars['Amplitude'] >= .05 ))]

    #ampBin1 = variableStars[np.where(variableStars['Amplitude'] <= .1)]
    ampBin2 = variableStars[np.where((variableStars['Amplitude'] <= .15) & (variableStars['Amplitude'] >= .1 ))]
    ampBin3 = variableStars[np.where((variableStars['Amplitude'] <= .2) & (variableStars['Amplitude'] >= .15 ))]
    ampBin4 = variableStars[np.where((variableStars['Amplitude'] <= .25) & (variableStars['Amplitude'] >= .2 ))]
    ampBin5 = variableStars[np.where((variableStars['Amplitude'] <= .3) & (variableStars['Amplitude'] >= .25 ))]
    ampBin6 = variableStars[np.where((variableStars['Amplitude'] <= .35) & (variableStars['Amplitude'] >= .3 ))]
    ampBin7 = variableStars[np.where((variableStars['Amplitude'] <= .4) & (variableStars['Amplitude'] >= .35 ))]
    ampBin8 = variableStars[np.where((variableStars['Amplitude'] <= .45) & (variableStars['Amplitude'] >= .4 ))]
    ampBin9 = variableStars[np.where((variableStars['Amplitude'] <= .5) & (variableStars['Amplitude'] >= .45 ))]
    ampBin10 = variableStars[np.where((variableStars['Amplitude'] <= .55) & (variableStars['Amplitude'] >= .5 ))]
    ampBin11 = variableStars[np.where((variableStars['Amplitude'] <= .6) & (variableStars['Amplitude'] >= .55 ))]
    ampBin12 = variableStars[np.where((variableStars['Amplitude'] <= .65) & (variableStars['Amplitude'] >= .6 ))]
    ampBin13 = variableStars[np.where((variableStars['Amplitude'] <= .7) & (variableStars['Amplitude'] >= .65 ))]
    ampBin14 = variableStars[np.where((variableStars['Amplitude'] <= .75) & (variableStars['Amplitude'] >= .7 ))]
    ampBin15 = variableStars[np.where((variableStars['Amplitude'] <= .8) & (variableStars['Amplitude'] >= .75 ))]

    ampBin16 = variableStars[np.where((variableStars['Amplitude'] <= .85) & (variableStars['Amplitude'] >= .8 ))]
    ampBin17 = variableStars[np.where((variableStars['Amplitude'] <= .9) & (variableStars['Amplitude'] >= .85 ))]
    ampBin18 = variableStars[np.where((variableStars['Amplitude'] <= .95) & (variableStars['Amplitude'] >= .9 ))]
    ampBin19 = variableStars[np.where((variableStars['Amplitude'] <= 1) & (variableStars['Amplitude'] >= .95 ))]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'\textbf{Catalina Variables}')
    plt.ylabel(r'\textbf{Amplitude}')
    plt.xlabel(r'\textbf{Sigma off Baseline Curve}')
    axes = plt.gca()
    axes.set_xlim([0,100])
    axes.set_ylim([0,2])

    plt.scatter(variableStars['SigmaOffBaseline'], variableStars['Amplitude'], s = .05, color = 'red', alpha=0.5)
    #plt.scatter(statistics.mean(ampBin0['SigmaOffBaseline']), statistics.mean(ampBin0['Amplitude']), s = 2, color = 'black' )

    plt.scatter(statistics.mean(ampBin1['SigmaOffBaseline']), statistics.mean(ampBin1['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin2['SigmaOffBaseline']), statistics.mean(ampBin2['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin3['SigmaOffBaseline']), statistics.mean(ampBin3['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin4['SigmaOffBaseline']), statistics.mean(ampBin4['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin5['SigmaOffBaseline']), statistics.mean(ampBin5['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin6['SigmaOffBaseline']), statistics.mean(ampBin6['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin7['SigmaOffBaseline']), statistics.mean(ampBin7['Amplitude']), s = 2, color = 'black' )
    plt.scatter(statistics.mean(ampBin8['SigmaOffBaseline']), statistics.mean(ampBin8['Amplitude']), s = 2, color = 'black')
    #plt.scatter(statistics.mean(ampBin9['SigmaOffBaseline']), statistics.mean(ampBin9['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin10['SigmaOffBaseline']), statistics.mean(ampBin10['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin11['SigmaOffBaseline']), statistics.mean(ampBin11['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin12['SigmaOffBaseline']), statistics.mean(ampBin12['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin13['SigmaOffBaseline']), statistics.mean(ampBin13['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin14['SigmaOffBaseline']), statistics.mean(ampBin14['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin15['SigmaOffBaseline']), statistics.mean(ampBin15['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin16['SigmaOffBaseline']), statistics.mean(ampBin16['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin17['SigmaOffBaseline']), statistics.mean(ampBin17['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin18['SigmaOffBaseline']), statistics.mean(ampBin18['Amplitude']), s = 1, color = 'black' )
    #plt.scatter(statistics.mean(ampBin19['SigmaOffBaseline']), statistics.mean(ampBin19['Amplitude']), s = 1, color = 'black' )

#    plt.scatter(variableStars['SigmaOffBaseline'], variableStars['Amplitude'], s = .2)
    print(statistics.mean(ampBin1['SigmaOffBaseline']))
    print(statistics.mean(ampBin2['SigmaOffBaseline']))
    print(statistics.mean(ampBin3['SigmaOffBaseline']))
    print(statistics.mean(ampBin4['SigmaOffBaseline']))
    print(statistics.mean(ampBin5['SigmaOffBaseline']))
    print(statistics.mean(ampBin6['SigmaOffBaseline']))
    print(statistics.mean(ampBin7['SigmaOffBaseline']))
    print(statistics.mean(ampBin8['SigmaOffBaseline']))
    #print(statistics.mean(ampBin9['SigmaOffBaseline']))
    #print(statistics.mean(ampBin10['SigmaOffBaseline']))
    #print(statistics.mean(ampBin11['SigmaOffBaseline']))
    #print(statistics.mean(ampBin12['SigmaOffBaseline']))
    #print(statistics.mean(ampBin13['SigmaOffBaseline']))
    #print(statistics.mean(ampBin14['SigmaOffBaseline']))
    #print(statistics.mean(ampBin15['SigmaOffBaseline']))

    print("----------------------")
    print(statistics.mean(ampBin1['Amplitude']))
    print(statistics.mean(ampBin2['Amplitude']))
    print(statistics.mean(ampBin3['Amplitude']))
    print(statistics.mean(ampBin4['Amplitude']))
    print(statistics.mean(ampBin5['Amplitude']))
    print(statistics.mean(ampBin6['Amplitude']))
    print(statistics.mean(ampBin7['Amplitude']))
    print(statistics.mean(ampBin8['Amplitude']))
    '''
    print(statistics.mean(ampBin9['Amplitude']))
    print(statistics.mean(ampBin10['Amplitude']))
    print(statistics.mean(ampBin11['Amplitude']))
    print(statistics.mean(ampBin12['Amplitude']))
    print(statistics.mean(ampBin13['Amplitude']))
    print(statistics.mean(ampBin14['Amplitude']))
    print(statistics.mean(ampBin15['Amplitude']))'''

    print("----------------------")

    print(statistics.stdev(ampBin1['Amplitude']))
    print(statistics.stdev(ampBin2['Amplitude']))
    print(statistics.stdev(ampBin3['Amplitude']))
    print(statistics.stdev(ampBin4['Amplitude']))
    print(statistics.stdev(ampBin5['Amplitude']))
    print(statistics.stdev(ampBin6['Amplitude']))
    print(statistics.stdev(ampBin7['Amplitude']))
    print(statistics.stdev(ampBin8['Amplitude']))
    '''
    print(statistics.stdev(ampBin9['Amplitude']))
    print(statistics.stdev(ampBin10['Amplitude']))
    print(statistics.stdev(ampBin11['Amplitude']))
    print(statistics.stdev(ampBin12['Amplitude']))
    print(statistics.stdev(ampBin13['Amplitude']))
    print(statistics.stdev(ampBin14['Amplitude']))
    print(statistics.stdev(ampBin15['Amplitude']))'''

    print("----------------------")

    print(statistics.stdev(ampBin1['SigmaOffBaseline']))
    print(statistics.stdev(ampBin2['SigmaOffBaseline']))
    print(statistics.stdev(ampBin3['SigmaOffBaseline']))
    print(statistics.stdev(ampBin4['SigmaOffBaseline']))
    print(statistics.stdev(ampBin5['SigmaOffBaseline']))
    print(statistics.stdev(ampBin6['SigmaOffBaseline']))
    print(statistics.stdev(ampBin7['SigmaOffBaseline']))
    print(statistics.stdev(ampBin8['SigmaOffBaseline']))
    '''
    print(statistics.stdev(ampBin9['SigmaOffBaseline']))
    print(statistics.stdev(ampBin10['SigmaOffBaseline']))
    print(statistics.stdev(ampBin11['SigmaOffBaseline']))
    print(statistics.stdev(ampBin12['SigmaOffBaseline']))
    print(statistics.stdev(ampBin13['SigmaOffBaseline']))
    print(statistics.stdev(ampBin14['SigmaOffBaseline']))
    print(statistics.stdev(ampBin15['SigmaOffBaseline']))'''


    X = np.array([statistics.mean(ampBin1['SigmaOffBaseline']), statistics.mean(ampBin2['SigmaOffBaseline']), statistics.mean(ampBin3['SigmaOffBaseline']), statistics.mean(ampBin4['SigmaOffBaseline']), statistics.mean(ampBin5['SigmaOffBaseline']), statistics.mean(ampBin6['SigmaOffBaseline']), statistics.mean(ampBin7['SigmaOffBaseline']), statistics.mean(ampBin8['SigmaOffBaseline'])]).reshape((-1, 1))
    Y = np.array([statistics.mean(ampBin1['Amplitude']), statistics.mean(ampBin2['Amplitude']), statistics.mean(ampBin3['Amplitude']), statistics.mean(ampBin4['Amplitude']), statistics.mean(ampBin5['Amplitude']), statistics.mean(ampBin6['Amplitude']), statistics.mean(ampBin7['Amplitude']), statistics.mean(ampBin8['Amplitude'])])
    #X = np.array([statistics.mean(ampBin1['SigmaOffBaseline']), statistics.mean(ampBin2['SigmaOffBaseline']), statistics.mean(ampBin3['SigmaOffBaseline']), statistics.mean(ampBin4['SigmaOffBaseline']), statistics.mean(ampBin5['SigmaOffBaseline']), statistics.mean(ampBin6['SigmaOffBaseline']), statistics.mean(ampBin7['SigmaOffBaseline']), statistics.mean(ampBin8['SigmaOffBaseline']), statistics.mean(ampBin9['SigmaOffBaseline']), statistics.mean(ampBin10['SigmaOffBaseline']), statistics.mean(ampBin11['SigmaOffBaseline']), statistics.mean(ampBin12['SigmaOffBaseline']), statistics.mean(ampBin13['SigmaOffBaseline']), statistics.mean(ampBin14['SigmaOffBaseline']), statistics.mean(ampBin15['SigmaOffBaseline'])]).reshape((-1, 1))
    #Y = np.array([statistics.mean(ampBin1['Amplitude']), statistics.mean(ampBin2['Amplitude']), statistics.mean(ampBin3['Amplitude']), statistics.mean(ampBin4['Amplitude']), statistics.mean(ampBin5['Amplitude']), statistics.mean(ampBin6['Amplitude']), statistics.mean(ampBin7['Amplitude']), statistics.mean(ampBin8['Amplitude']), statistics.mean(ampBin9['Amplitude']), statistics.mean(ampBin10['Amplitude']), statistics.mean(ampBin11['Amplitude']), statistics.mean(ampBin12['Amplitude']), statistics.mean(ampBin13['Amplitude']), statistics.mean(ampBin14['Amplitude']), statistics.mean(ampBin15['Amplitude'])])
    model = LinearRegression()
    model.fit(X, Y)
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    plt.plot(X, Y_pred, color='black', linewidth=1)

    '''
    regr = linear_model.LinearRegression()
    regr.fit(Xdata, range(len(X)))
    Y_pred = regr.predict(Xdata)
    plt.plot(Xdata, Y_pred, color='black', linewidth=1)

    Xdata = []
    X = [statistics.mean(ampBin1['SigmaOffBaseline']), statistics.mean(ampBin2['SigmaOffBaseline']), statistics.mean(ampBin3['SigmaOffBaseline']), statistics.mean(ampBin4['SigmaOffBaseline']), statistics.mean(ampBin5['SigmaOffBaseline']), statistics.mean(ampBin6['SigmaOffBaseline']), statistics.mean(ampBin7['SigmaOffBaseline']), statistics.mean(ampBin8['SigmaOffBaseline'])]
    Y = [statistics.mean(ampBin1['Amplitude']), statistics.mean(ampBin2['Amplitude']), statistics.mean(ampBin3['Amplitude']), statistics.mean(ampBin4['Amplitude']), statistics.mean(ampBin5['Amplitude']), statistics.mean(ampBin6['Amplitude']), statistics.mean(ampBin7['Amplitude']), statistics.mean(ampBin8['Amplitude'])]
    for i in range(len(X)):
        Xdata.append([X[i], [Y[i]]])
    '''


    plt.show()



# function that takes in relevant data of known variable star
# flags stars based on how many sigma they are from mean magnitude error of given photometric and magnitude bin
# returns # of sigma variable star is from mean
def getSigma(magnitude, magError, numObs):

    ## get Gaussian Data for different photometric observation numbers

    sigma = -1000

    if numObs > 50 and numObs < 100:
        sigma = evaluateBounds(getData('50-100.csv'), magError, magnitude)

    elif numObs < 150:
        sigma = evaluateBounds(getData('100-150.csv'), magError, magnitude)

    elif numObs < 200:
        sigma = evaluateBounds(getData('150-200.csv'), magError, magnitude)

    elif numObs < 250:
        sigma = evaluateBounds(getData('200-250.csv'), magError, magnitude)

    elif numObs < 300:
        sigma = evaluateBounds(getData('250-300.csv'), magError, magnitude)

    elif numObs < 350:
        sigma = evaluateBounds(getData('300-350.csv'), magError, magnitude)

    elif numObs < 400:
        sigma = evaluateBounds(getData('350-400.csv'), magError, magnitude)

    elif numObs < 450:
        sigma = evaluateBounds(getData('400-450.csv'), magError, magnitude)

    elif numObs < 500:
        sigma = evaluateBounds(getData('450-500.csv'), magError, magnitude)

    elif numObs < 550:
        sigma = evaluateBounds(getData('500-550.csv'), magError, magnitude)

    elif numObs < 600:
        sigma = evaluateBounds(getData('550-600.csv'), magError, magnitude)

    elif numObs < 650:
        sigma = evaluateBounds(getData('600-650.csv'), magError, magnitude)

    elif numObs < 700:
        sigma = evaluateBounds(getData('650-700.csv'), magError, magnitude)

    else:
        sigma = evaluateBounds(getData('700-750.csv'), magError, magnitude)

    return sigma



def evaluateBounds(GaussianData, magError, magnitude):
    ## get magnitude bin in Gaussian fit that is closest to magnitude of star
    min = 100
    desiredBin = 0
    for datapoint in range(len(GaussianData['Magnitude_of_Bin'])):
        if abs(GaussianData['Magnitude_of_Bin'][datapoint]-magnitude) < min:
            min = abs(GaussianData['Magnitude_of_Bin'][datapoint]-magnitude)
            desiredBin = datapoint

    meanMagError = GaussianData['MeanMagError'][desiredBin]

    difference = magError - meanMagError #only considering upper bounds; to include stars of more than 1 sigma below mean, use abs

    # return the number of standard deviations the magnitude error is from the mean of the given magnitude bin
    return difference/GaussianData['StdevMagError'][desiredBin]



# a function that gets Gaussian Data from CSV file of specified photometric count range
def getData(photoObsRange):
    dtype = [('MeanMagError', float), ('StdevMagError', float), ('UpperBoundMagError', float), ('Magnitude_of_Bin', float)]
    values = []

    filename = '/Users/touatokuchi/Desktop/MSU/KnownVariablesTest/' + photoObsRange
    with open(filename) as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            try:
                meanMagError = float(row[0])
                stdevMagError = float(row[1])
                UpperBoundMagError = float(row[2])
                Magnitude_of_Bin = float(row[3])
                values.append((meanMagError, stdevMagError, UpperBoundMagError, Magnitude_of_Bin))

            except ValueError:
                continue

    PhotoRangeData = np.array(values, dtype=dtype)
    return PhotoRangeData



def main():
    plotCatalinaUncertainties()

if __name__== "__main__":
    main()
