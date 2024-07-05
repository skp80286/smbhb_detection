import lightcurvesimulation as LC
import numpy as np
import random
from dataclasses import dataclass
from PyAstronomy.pyTiming import pyPeriod
from astropy.timeseries import LombScargle
import scipy.signal as sig 
import pandas as pd
import time
import math
import sys
from scipy.stats import binned_statistic

@dataclass
class GlsAnalysisParams:
    is_binary = True
    survey = LC.LSST
    baseline = survey.observation_period
    tau = baseline * 10
    sigma = np.exp(-1.6)
    amplitude = 0.5
    period= 1000
    phase = 0
    package_to_use = 'pyastronomy'
    smooth_observations = None # If not None, this indicates the number days of observations to be combined for a smooth curve
    norm = 'chisq' # “ZK”, “Scargle”, “HorneBaliunas”, “Cumming”, “wrms”, “chisq”.
    
    def __str__(self):
        return 'sigma={0:.2f},amplitude={1:.2f}'.format(self.sigma, self.amplitude)

class GlsAnalysisRunner:
    # Use binned_statistic to calculate mean within each bin
    def get_smooth_curve(self, num_bins, t, mag, magerr):
        bin_mag, bin_edges, binnumber = binned_statistic(t, mag, statistic=np.nanmean, bins=num_bins)
        bin_magerr, bin_edges, binnumber = binned_statistic(t, magerr, statistic=np.nanmean, bins=num_bins)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_t = bin_edges[1:] - bin_width/2
    
        filter = np.isfinite(bin_mag)
        #print(str(filter[:10]))
        bin_mag = bin_mag[filter]
        bin_t = bin_t[filter]
        bin_magerr = bin_magerr[filter]
        return (bin_t, bin_mag, bin_magerr)
    
#plt.scatter(bin_t, bin_mag)
#plt.show()
    def run(self, params = GlsAnalysisParams(), num_iterations = 100):
        lcsim = LC.LightCurveSimulation()
        periods = np.arange(30, 3000, 10) 
        freqs = 1./periods 
        experiment_data = []
        start_time = time.time()
        for _ in range(num_iterations):
            is_binary = bool(random.getrandbits(1))
            period = None
            #print("[AnalysisRunner] is_binary: " + str(is_binary))
            if is_binary:
                period = random.randint(30, 3000)
                #print("[AnalysisRunner] period: " + str(period))

            (t, mag, magerr, _, _, _) = lcsim.generate_light_curve(
                survey=params.survey, is_binary = is_binary, tau=params.tau, 
                sigma=params.sigma, amplitude=params.amplitude, period=period, phase=params.phase)

            if params.smooth_observations is not None:
                (t, mag, magerr) = self.get_smooth_curve(params.survey.observation_period/params.smooth_observations, t, mag, magerr)
            # Perform analysis
            # Compute the GLS periodogram with default options.
            power = None
            if params.package_to_use == 'pyastronomy':            
                clp = pyPeriod.Gls((t, mag, magerr), freq=freqs, norm=params.norm) 
                power = clp.power
            elif params.package_to_use == 'scipy': 
                power = sig.lombscargle(t, mag, freqs)
            else: #astroPy
                ls = LombScargle(t , mag, magerr)
                power = ls.power(freqs)

            # Find the frequencies with the strongest signal
            # get the max power
            top_ind = np.argmax(power)
            detected_period = periods[top_ind]
            max_power = power[top_ind]
            #

            max_fap_adj = None
            if params.package_to_use == 'pyastronomy':            
                max_fap = clp.FAP(max_power)
            elif params.package_to_use == 'astroPy': 
                max_fap = ls.false_alarm_probability(max_power)
            if max_fap > 0:
                max_fap_adj = -(math.log10(max_fap))
            else:
                max_fap_adj = sys.float_info.min
            period_error= None
            if is_binary:
                period_error = abs(detected_period/period - 1)

            peak_ind, _ = sig.find_peaks(power, max_power/4.0)
            num_peaks = len(peak_ind)
            experiment_data.append([is_binary, period, detected_period, period_error, max_power, max_fap_adj, num_peaks])
            
        print('Execution time: {0:.2f}'.format(time.time() - start_time))

        return experiment_data


