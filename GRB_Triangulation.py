import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

from sgp4.api import jday
from sgp4.api import Satrec
from astropy.coordinates import TEME, CartesianRepresentation, GCRS
from astropy import units as u
from astropy import constants as const
from astropy.time import Time
from spacetrack import SpaceTrackClient

from pyipn.correlation import correlate
from pyipn.detector import Detector
from pyipn.utils.timing import *
from pyipn.geometry import DetectorLocation

from Trigreader import TrigReader

class Integral:
    """Class to save Integral data needed for triangulation."""

    def __init__(self, filename, tle_filename, bkg_neg_stop, bkg_pos_start, fermi_trigsecond):
        """
        :param filename: path of the Integral SPI-ACS file to be read
        :param tle_filename: path of txt-file with twoline-elements of Integral, if None: TLE is downloaded
        :param bkg_neg_stop: parameter from Balrog, end of background fit before Fermi triggertime
        :param bkg_pos_start: parameter from Balrog, start of background fit after Fermi triggertime
        :param fermi_trigsecond: second of the day of corresponding Fermi trigger
        """
        self.filename = filename
        self.tle_filename = tle_filename #path of file with all integral tles 
        self.triggertime, self.secondOfDay, self.date = self.getTriggertime() #triggertime as julian Date, second of day and date
        
        #GRB start and end from Fermi -> correction by triggertime difference
        self.GRB_start = bkg_neg_stop + self.secondOfDay - fermi_trigsecond
        self.GRB_end = bkg_pos_start + self.secondOfDay - fermi_trigsecond

        #handle: GRB_start before data starts
        if self.GRB_start < -5:
            self.GRB_start = -5
        
        self.lightcurve = self.getLightcurve() #raw data
        self.BGsubLightcurve = self.getBGsubLightcurve() #background-subtracted data, background fitted by constant function in the intervals [-5s, GRB_start] and [GRB_end, 100s]
        
        if tle_filename == None:
            self.tle1, self.tle2 = self.downloadTLE() #download tle from spacetrack.org if no tle_file is given
        else:
            self.tle1, self.tle2 = self.getTLEfromtxt() #most recent tle from tle_file in respect to triggertime

        self.binsize = float(pd.read_csv(self.filename, delim_whitespace=True, header = None, nrows=2).iloc[1][1])
        self.times = np.array(self.lightcurve["time"]) + self.secondOfDay #absolute time in seconds of the day
        self.cnts_error = np.array(self.lightcurve["rate"])*0.05 
        self.position = self.getPosition() #integral position at triggertime from tle

    def getLightcurve(self):
        data = pd.read_csv(self.filename, delim_whitespace=True , header = None, skiprows=2)
        data.columns = ("time", "rate")
        return data
    
    def getBGsubLightcurve(self):
        backgroundFit_data = self.lightcurve.drop(self.lightcurve[(self.GRB_start < self.lightcurve["time"]) & (self.lightcurve["time"] < self.GRB_end)].index)
        def func(x,a):
            return a
        popt, pcov = curve_fit(func, backgroundFit_data["time"], backgroundFit_data["rate"])
        tmp = self.lightcurve.copy()
        tmp["rate"] -= func(tmp["time"],*popt)
        return tmp
    
    def getTriggertime(self):
        header = pd.read_csv(self.filename, delim_whitespace=True, header = None, nrows=2)
        date = header.iloc[0][2].replace("'","").split("/")
        secondOfDay = header.iloc[0][3]
        jd, fr = jday(int(date[2])+2000,int(date[1]),int(date[0]),0,0,0)
        assert fr == 0
        return jd + secondOfDay/(24*3600), secondOfDay, date
    
    def getTLEfromtxt(self):
        tles = pd.read_csv(self.tle_filename, header=None)[0].tolist()
        tle_date = int(self.date[2])*1000 + datetime.date(int(self.date[2])+2000, int(self.date[1]), int(self.date[0])).timetuple().tm_yday + (self.triggertime - 0.5)%1
        for i in range(int(len(tles)//2)):
            if float(tles[2*i].split()[3]) > tle_date:
                return tles[2*i-2], tles[2*i-1]

    def downloadTLE(self):
        tle = SpaceTrackClient("ludwig.schmidt@tum.de","GRBBachelor2022").tle(norad_cat_id=27540, 
        epoch=f"<{self.date[2]}-{self.date[1]}-{self.date[0]}", orderby = "epoch desc", limit = 1, format = "tle").split("\n")
        return tle[0], tle[1]


    def getPosition(self):
        #returns satellite position (GCRS) from TLE and julianDate
        satellite = Satrec.twoline2rv(self.tle1,self.tle2)
        e,teme_p,teme_v = satellite.sgp4(self.triggertime, 0)
        assert e == 0
        t = Time(self.triggertime, format="jd")
        return TEME(CartesianRepresentation(teme_p*u.km), obstime=t).transform_to(GCRS(obstime = t))


class Fermi:
    """Class to save Fermi Trigdat data needed for triangulation."""

    def __init__(self, filename, minRes, bkg_neg_stop, bkg_pos_start, date, fermi_trigsecond, channel_range = (0,7)): 
        """
        :param filename: path of the Fermi Trigdat file to be read (.fit)
        :param minRes: lowest time resolved data to be used, possible values: 0.064, 0.256, 1.024, 8.192
        :param bkg_neg_stop: parameter from Balrog, end of background fit before Fermi triggertime
        :param bkg_pos_start: parameter from Balrog, start of background fit after Fermi triggertime
        :param date: date of the event as string yyyymmdd
        :param fermi_trigsecond: triggertime as second of the day
        :param channel_range: tuple with lowest and highest BGO energy channel to be used, all channels = (0,7)
        """
        self.filename = filename
        self.GRB_start = bkg_neg_stop
        self.GRB_end = bkg_pos_start
        self.channel_range = channel_range
        self.minRes = minRes
        self.date = date
        self.secondOfDay = fermi_trigsecond

        self.triggertime = jday(int(self.date[0:4]), int(self.date[4:6]), int(self.date[6:8]), 0, 0, 0)[0] + self.secondOfDay/(24*3600) #triggertime as julian date

        self.lightcurves, self.binsize = self.getBGOLightcurves() #list of Dataframes (time, rate), best results only using BGO detectors
        self.BGsubLightcurves = self.getBGsubLightcurves(self.lightcurves) #background-subtracted lightcurves, background fitted by polynomial (3rd degree)
        self.lightcurveIndex = self.mostIntensiveLightcurve(self.BGsubLightcurves) #use more intensive BGO lightcurve (after background subtraction)
        self.lightcurve = self.rebinLightcurve(self.lightcurves[self.lightcurveIndex]) #rebin all timebins to 64ms
        self.BGsubLightcurve = self.rebinLightcurve(self.BGsubLightcurves[self.lightcurveIndex])
        
        
        self.times = np.array(self.lightcurve["time"]) + self.secondOfDay
        self.cnts_error = np.array(self.lightcurve["rate"])*0.064
        self.position = self.getPosition() #position at triggertime form fermi trigdat data
    

    def getBGOLightcurves(self):
        trig = TrigReader(self.filename, fine = True)
        lightcurves = []
        tstart = trig._tstart
        tstop = trig._tstop
        rates = trig._rates
        rates = np.sum(rates[:,:,self.channel_range[0]:self.channel_range[1]+1], axis = 2) #sum couts of selcted energy channels
        for det in range(12,14): #indices of BGO detectors
            lightcurves.append(pd.DataFrame({"time": tstart, "rate": rates[:,det]}))
        return lightcurves, tstop - tstart

    def getBGsubLightcurves(self, lightcurves):
        BGsubLightcurves = []
        for curve in lightcurves:
            tmp = curve.copy()
            backgroundFit_data = curve.drop(curve[(curve["time"] > self.GRB_start) & (curve["time"] < self.GRB_end)].index)
            def func(x,a,b,c,d):
                return a*x**3+b*x**2+c*x+d
            popt, pcov = curve_fit(func, backgroundFit_data["time"], backgroundFit_data["rate"])
            tmp["rate"] -= func(tmp["time"],*popt)
            BGsubLightcurves.append(tmp)
        return BGsubLightcurves

    def mostIntensiveLightcurve(self, lightcurves):
        count_sum = []
        for lightcurve in lightcurves:
            count_sum.append(np.sum(lightcurve["rate"][25:38]*self.binsize[25:38]))
        return count_sum.index(max(count_sum))
        
    def rebinLightcurve(self, lightcurve):
        assert self.minRes in (0.064,0.256,1.024,8.192)
        indices = np.argwhere(self.binsize < self.minRes + 0.01).flatten()
        arr_nBins = np.around(self.binsize[indices]/0.064, 0).astype(int)
        rebinnedLightcurve = pd.DataFrame(np.repeat(lightcurve.values[indices], arr_nBins, axis=0))
        rebinnedLightcurve.columns = ("time", "rate")
        i = 0
        for bins in arr_nBins:
           for j in range(bins):
            rebinnedLightcurve["time"][i] += j*0.064
            i +=1
        return rebinnedLightcurve

    def getPosition(self):
        trig = TrigReader(self.filename, fine = True)
        position = trig._sc_pos[np.argwhere(trig._tstart== min(abs(trig._tstart)))].flatten()
        assert len(position) == 3, "Fermi position could not be read from Trigdat file."
        t = Time(self.triggertime, format = "jd")
        return TEME(CartesianRepresentation(position*u.m),obstime=t).transform_to(GCRS(obstime=t))

def triangulate(integral, fermi, nSigma=1, fit=True, fit_range=(-2,3)):
    """
    Function to determine GRB position by triangulation from Integral and Fermi object.
    :param integral: Integral object
    :param fermi: Fermi object
    :param balrog: Balrog object, if None: no Balrog localization shown in annulus plot
    :param nSigma: confidence level of theta error
    :param fit: if True: fit around minimum of cross-correlation function with parabola
    :param fit_range: range of data points around minimum to be fitted

    :returns: matplotlib.figure objects of cross-correlation and annulus plot, pointing of connecting vector between fermi and integral (right ascension/declination), opening angle theta of triangulation circle with error 
    """

    assert abs(integral.secondOfDay - fermi.secondOfDay) < 10, "Trigger times too far apart."
    
    posInt = integral.position
    posFer = fermi.position 
    
    #create pyipn.detector.Detector objects for Integral and Fermi
    d1 = Detector(DetectorLocation(posInt.ra.degree, posInt.dec.degree, posInt.distance - 6371 * u.km, posInt.obstime), None, None, "Integral") #pointing and effective area not required

    d2 = Detector(DetectorLocation(posFer.ra.degree, posFer.dec.degree, posFer.distance - 6371 * u.km, posFer.obstime), None, None, "Fermi")

    #calculate detector distance and pointing of connecting vector
    distance, norm_d, ra, dec = calculate_distance_and_norm(d2,d1)


    #create data needed for pyipn.correlation.correlate function
    bgsubcnts1 = np.array(integral.BGsubLightcurve["rate"])*0.05
    bgsubcnts2 = np.array(fermi.BGsubLightcurve["rate"])*0.064
    
    assert len(bgsubcnts1) > 2000, "Integral SPI-ACS data incomplete."
    
    #use Integral data from -5s to 50s relative to triggertime
    i_beg_1 = 0
    n_max_1 = np.argwhere(np.array(integral.BGsubLightcurve["time"]) < 50).flatten()[-1]
    
    #start and end index of Fermi lightcurve to be used 
    i_beg_2 = 0
    i_end_2 = len(bgsubcnts2)-1
    if fermi.minRes == 1.024:
        i_beg_2 = np.argwhere(np.array(fermi.BGsubLightcurve["time"]) > -2 + integral.secondOfDay - fermi.secondOfDay).flatten()[0] #lc2 start at -2 s -> lc2 must not start before lc1
        if fermi.GRB_end < 40:
            i_end_2 = np.argwhere(np.array(fermi.BGsubLightcurve["time"]) < fermi.GRB_end).flatten()[-1] #lc2 end at GRB_end
        else:
            i_end_2 = np.argwhere(np.array(fermi.BGsubLightcurve["time"]) < 40).flatten()[-1]

    i1, i2 = get_max_sn((i_end_2-i_beg_2)*0.064, bgsubcnts1) #indices of lc1 corresponding to i_beg_1 and i_beg_2 for calculating fscale
    
    fscale = (np.sum(bgsubcnts1[i1:i2+1]))/(np.sum(bgsubcnts2[i_beg_2:i_end_2+1]))
    
    #fscale = max(bgsub1)/max(bgsub2)
    assert fscale > 0, "fscale must be positive."

    arr_dt, arr_chi, nDOF, fRijMin, dTmin, iMin, nMin = correlate(integral.times, bgsubcnts1, integral.cnts_error, fermi.times, bgsubcnts2, fermi.cnts_error,
    i_beg_2, i_end_2, i_beg_1, n_max_1, fscale, integral.binsize*1000, 64)
    
    #search for minimum of correlation function in plausible interval
    maxTimelag = float((distance.to("m")/const.c)/u.s)
    possibleTimelags = np.argwhere(abs(arr_dt) < maxTimelag) #empty for large interval between trigger times, maybe solved for correct GRB_start/GRB_end

    nMin = int(np.argwhere(arr_chi[possibleTimelags].flatten() == min(arr_chi[possibleTimelags].flatten()))) + int(possibleTimelags[0])
    dTmin = arr_dt[nMin]

    #create cross-correlation plot
    crosscor_fig, crosscor_ax = plt.subplots()
    crosscor_ax.plot(arr_dt, arr_chi, markevery = [nMin], marker = "d")
    crosscor_ax.axvline(-maxTimelag, color = "k", linestyle = "dashed", linewidth = 1)
    crosscor_ax.axvline(maxTimelag, color = "k", linestyle = "dashed", linewidth = 1)
    crosscor_ax.set_xlabel(r"Timelag: Integral $\rightarrow$ Fermi (s)")
    crosscor_ax.set_ylabel(r"Cross-correlation $r_{ij}^2$")
    crosscor_ax.set_xlim(-3,3)
    
    #fit around minimum with parabola
    if fit==True:
        def func(x,a,b,c):
            return a*x**2+b*x+c
        popt, pcov = curve_fit(func, arr_dt[nMin+fit_range[0]:nMin+fit_range[1]], arr_chi[nMin+fit_range[0]:nMin+fit_range[1]])
        x = np.linspace(arr_dt[nMin+fit_range[0]], arr_dt[nMin+fit_range[1]-1])
        crosscor_ax.plot(x,func(x,*popt))
        dt = -popt[1]/(2*popt[0])*u.s
        
    else:
        dt = dTmin*u.s
    
    #if minimum of parabola outside plausible interval: use minimum of crosscor function
    if dt > maxTimelag*u.s or dt < -maxTimelag*u.s:
        dt = dTmin*u.s
    
    #index closest to dt
    nMin = np.argmin(abs(arr_dt - dt/u.s))

    #compute errors
    dTupper_cons, dTlower_cons, dTupper_interp, dTlower_interp, fsigma = timelag_error(nSigma, nDOF, arr_chi, arr_dt, nMin, maxTimelag)

    
    skw_dict = create_skw_dict("astro degrees mollweide", None, None)
    annulus_fig, annulus_ax = plt.subplots(subplot_kw=skw_dict)
    #annulus_ax.set_xlabel("Right Ascension")
    #annulus_ax.set_ylabel("Declination")
    
    
    #triangulation circle
    compute_annulus_from_time_delay(dt, distance/const.c, d1, d2, ax = annulus_ax, color = "black")
    
    #conservative error
    compute_annulus_from_time_delay(dTupper_cons*u.s, dTlower_cons*u.s, d1, d2, ax = annulus_ax)

    #interpolated error
    compute_annulus_from_time_delay(dTupper_interp*u.s, dTlower_interp*u.s, d1, d2, ax = annulus_ax, linestyle = "dashed")

    annulus_ax.set_title(f"Triangulation annulus with {nSigma}\u03C3-errors \n (dashed errors from linear interpolation)")
    annulus_ax.grid()
    annulus_ax.legend()

    
    #distinguish cases so that 0°<theta<90°
    if dt > 0:
        distance, norm_d, ra, dec = calculate_distance_and_norm(d2,d1)
        return crosscor_fig, annulus_fig, ra.degree, dec.degree, theta_from_time_delay(dt, distance)*180/np.pi, theta_from_time_delay(dTlower_cons*u.s, distance)*180/np.pi, theta_from_time_delay(dTupper_cons*u.s, distance)*180/np.pi, theta_from_time_delay(dTlower_interp*u.s, distance)*180/np.pi, theta_from_time_delay(dTupper_interp*u.s, distance)*180/np.pi
    
    if dt < 0:
        distance, norm_d, ra, dec = calculate_distance_and_norm(d1,d2)
        return crosscor_fig, annulus_fig, ra.degree, dec.degree, theta_from_time_delay(-dt, distance)*180/np.pi, theta_from_time_delay(-dTupper_cons*u.s, distance)*180/np.pi, theta_from_time_delay(-dTlower_cons*u.s, distance)*180/np.pi, theta_from_time_delay(-dTupper_interp*u.s, distance)*180/np.pi, theta_from_time_delay(-dTlower_interp*u.s, distance)*180/np.pi


def timelag_error(nSigma, nDOF, arr_chi, arr_dt, nMin, maxTimelag):
    """
    From pyipn.correlation.Correlator._get_dTcc
    Computes confidence interval of determined timelag.
    :param nSigma: confidence level
    :param nDOF: output of pyipn.correlation.correlate
    :param arr_chi: output of pyipn.correlation.correlate
    :param arr_dt: output of pyipn.correlation.correlate
    :param nMin: index of minimal arr_chi
    """
    P0 = norm.cdf(nSigma) - norm.cdf(-nSigma)
    fSigma = chi2.ppf(P0, nDOF - 1) / (nDOF - 1) - 1.0 + arr_chi[nMin]
    #fSigmaSimple = arr_chi[nMin] + nSigma ** 2 /nDOF
    n = arr_chi.size

    # search upper dTcc 3 sigma
    i = nMin
    fRij = arr_chi[i]

    while (i > 1) and (fRij < fSigma):
        i = i - 1
        fRij = arr_chi[i]

    #conservative error: first data point after exceeding fsigma
    dTupper_cons = arr_dt[i]

    #arr_dt[i+1] < fsigma < arr_dt[i] -> interpolate by straight and find intersection with y = fsigma (interpolated error)
    if dTupper_cons < maxTimelag:
        dTupper_interp = (arr_dt[i] - arr_dt[i+1])*(fSigma - arr_chi[i])/(arr_chi[i] - arr_chi[i+1]) + arr_dt[i]
    else:
        dTupper_cons = maxTimelag
        dTupper_interp = maxTimelag

    # search lower dTcc 3 sigma
    i = nMin
    fRij = arr_chi[i]
    while (i < n - 1) and (fRij < fSigma):
        i = i + 1
        fRij = arr_chi[i]

    dTlower_cons = arr_dt[i]
    
    if dTlower_cons > - maxTimelag:
        dTlower_interp = (arr_dt[i-1] - arr_dt[i])*(fSigma - arr_chi[i])/(arr_chi[i-1] - arr_chi[i]) + arr_dt[i]
    else:
        dTlower_cons = - maxTimelag
        dTlower_interp = - maxTimelag
        
    return dTupper_cons, dTlower_cons, dTupper_interp, dTlower_interp, fSigma


def get_max_sn(dt_s, BGsubCounts):
    """
    From D. Svinkn/ pyipn.lightcurve.BinnedLightCurve.get_max_sn adapted
    :param dt_s: length of lightcurve 2 used in correlate() in seconds
    :param BGsubCounts: background subtracted counts of lightcurve 1
    :returns: start and end index of lightcurve 1 for calculating fscale 
    """

    len_int = int(dt_s // 0.05)

    cs = np.cumsum(BGsubCounts)

    fmax = 0.0
    i1 = 0
    i2 = cs.size
    for i in range(cs.size - len_int):
        dif = cs[i + len_int] - cs[i]
        if dif > fmax:
            fmax = dif
            i1 = i
            i2 = i + len_int

    return i1, i2


def autoTriangulate(acs_file, trigdat_file, bkg_neg_stop, bkg_pos_start, trigger_timestamp):
    """
    Function to automate triangulation.
    :param acs_file: path of Integral SPI-ACS file
    :param trigdat_file: path of Fermi Trigdat file
    :param bkg_neg_stop: from Balrog
    :param bkg_pos_start: from Balrog
    :param trigger_timestamp: from Balrog
    :returns: Matplotlib figures (lightcurve, cross-correlation, annulus) and parameters of the annulus (right ascension, declination, theta with confidence interval)
    """
    assert bkg_neg_stop < 0 and bkg_pos_start > 0, "bkg_neg_stop or bkg_pos_start with wrong sign."
    
    #read trigger_timestamp
    trigger_timestamp = trigger_timestamp.replace("Z", "").split("T")
    date = str(trigger_timestamp[0].replace("-",""))
    time = trigger_timestamp[1].split(":")
    trigsecond = int(time[0])*3600 + int(time[1])*60 + float(time[2])
    
    #create Integral object
    integral = Integral(acs_file, None, bkg_neg_stop, bkg_pos_start, trigsecond)
    
    #position of integral maximum relative to Fermi triggertime
    maxPos = integral.BGsubLightcurve["time"][np.argmax(integral.BGsubLightcurve["rate"])] + integral.secondOfDay - trigsecond
    
    #determine minRes
    if -0.25 < maxPos < 0.6:
        minRes = 0.064
    elif -1 < maxPos < 2:
        minRes = 0.256
    else:
        minRes = 1.024

    #create Fermi object
    fermi = Fermi(trigdat_file, minRes, bkg_neg_stop, bkg_pos_start, date, trigsecond)
    

    #triangulate
    crosscor_fig, annulus_fig, ra, dec, theta, dthetaUpper_cons, dthetaLower_cons, dthetaUpper_interp, dthetaLower_interp = triangulate(integral, fermi)
    
    #create lightcurve plot
    lightcurve_fig, lightcurve_ax = plt.subplots()
    lightcurve_ax.set_title("Lightcurves with trigger time (dashed)")
    lightcurve_ax.plot(integral.times - fermi.secondOfDay, integral.BGsubLightcurve["rate"], label = "Integral SPI-ACS")
    lightcurve_ax.plot(fermi.BGsubLightcurve["time"], fermi.BGsubLightcurve["rate"], label = "Fermi Trigdat")
    lightcurve_ax.set_xlabel(f"Time relative to Fermi trigger at {time[0]}:{time[1]}:{time[2][:5]} on {date[6:8]}/{date[4:6]}/{date[0:4]} (s)")
    lightcurve_ax.set_ylabel(r"Background subtracted count rate $\left(\frac{1}{s}\right)$")
    xlim_dict = {0.064: (- 0.5, + 1), 0.256: (- 1.5, + 2.5), 1.024: (- 2, + 40)}
    lightcurve_ax.set_xlim(xlim_dict[minRes])
    lightcurve_ax.legend()
    lightcurve_ax.axvline(integral.secondOfDay - fermi.secondOfDay, color = "#1f77b4", linestyle = "dashed", linewidth = 1)
    lightcurve_ax.axvline(0, color = "#ff7f0e", linestyle = "dashed", linewidth = 1)
    
    return lightcurve_fig, crosscor_fig, annulus_fig, ra, dec, theta, dthetaUpper_cons, dthetaLower_cons, dthetaUpper_interp, dthetaLower_interp
