import lightcurvesimulation as LC
import numpy as np
import random
from dataclasses import dataclass
from wwz import WWZ
from wwzplotter import WWZPlotter
import scipy.signal as sig 
import pandas as pd
import time
import math
import sys

@dataclass
class WwzAnalysisParams:
    is_binary = True
    survey = LC.LSST
    baseline = survey.observation_period
    tau = baseline * 10
    sigma = np.exp(-1.6)
    amplitude = 0.5
    period= 1000
    phase = 0

    def __str__(self):
        return 'sigma={0:.2f},amplitude={1:.2f}'.format(self.sigma, self.amplitude)

class WwzAnalysisRunner:
    def run(self, params = WwzAnalysisParams(), num_iterations = 100, verbose=False):
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
            
            # Perform analysis
            # Compute the WWZ transform.
            
            wwz = WWZ(verbose=False)
            wwz.set_data(t, mag, magerr, verbose=False)
            tau = wwz.get_tau(n_bins=10, verbose=False)
            wwz.set_freq(freqs, verbose=False)
            wwz.set_tau(tau, verbose=False)

            wwz.transform(snr_weights=True, verbose=0)
            i, j = np.unravel_index(np.argmax(wwz.wwz), wwz.wwz.shape)
            # Find the frequencies with the strongest signal
            # get the max power  
            detected_period = periods[j]
            max_power = wwz.wwz[i, j]
            #
            max_fap_adj = None
            period_error= None
            if is_binary:
                period_error = abs(detected_period/period - 1)

            if verbose:
                print('result: shape={0:} i={1:} j={2:} period={3:} detected_period={4:} max_power={5:}'
                      .format(wwz.wwz.shape, i, j, period, detected_period, max_power)) 

            num_peaks = None
            experiment_data.append([is_binary, period, detected_period, period_error, max_power, max_fap_adj, num_peaks])
            
        print('Execution time: {0:.2f}'.format(time.time() - start_time))

        return experiment_data


