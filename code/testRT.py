#Format data file

import numpy as np
import os
import MulensModel as mm
import RTModel as rt


def fluxtomag(mag, zeropint = 27.615):
    return -2.5*np.log10(mag) + zeropint

fu0 = -0.010
falpha = 54.0
alpha = 360.0 - falpha
file_name = "../Multiple_lc_run2-Me/s20p2s30p9psi60.0alpha%.1f/u%.3falpha%d"%(falpha,fu0,alpha)


#Read binary data
hjd, binaryflux, binaryflux_err = np.loadtxt(file_name+"_binflux.txt", usecols=[0,1,2], unpack=True)
my_data_bin = mm.MulensData([hjd, binaryflux, binaryflux_err], phot_fmt="flux", chi2_fmt="flux")
true_params = np.loadtxt(file_name+"_bininfo.txt", skiprows=1)
params_bin = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s": true_params[3], "q": true_params[4], "alpha": true_params[5], "rho": true_params[6]}

#Read planet data
hjd, planetflux, planetflux_err = np.loadtxt(file_name+"_planetflux.txt", usecols=[0,1,2], unpack=True)
my_data_planet = mm.MulensData([hjd, planetflux, planetflux_err], phot_fmt="flux", chi2_fmt="flux")
true_params = np.loadtxt(file_name+"_planetinfo.txt", skiprows=1)
params_planet = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s": true_params[3], "q": true_params[4], "alpha": true_params[5], "rho": true_params[6]}

true_chisq = 0.

#convert flux to magnitudes
binarymag = fluxtomag(binaryflux)
planetmag = fluxtomag(planetflux)
binarymag_err = 2.5*binaryflux_err/binaryflux
planetmag_err = 2.5*planetflux_err/planetflux


t_0 = params_bin['t_0']
u_0 = params_bin['u_0']
t_E = params_bin['t_E']
rho = params_bin['rho']
sbin = params_bin['s']
qbin = params_bin['q']
alphabin = 360. - params_bin['alpha'] #convert to alpha triplelens

sgns = np.random.choice([-1,1], size = 7)
params_bininit = {'t_0': 2002, 'u_0': -0.05, 't_E': 32., 's': 0.3, 'q': 0.7, 'alpha': 250., 'rho': 0.001            }#{'t_0': t_0 + sgns[0]*0.25, 'u_0': u_0 + sgns[1]*0.025*u_0, 't_E': t_E + sgns[2]*0.025*t_E, 's': sbin + sgns[3]*0.025*sbin, 'q': qbin + sgns[4]*0.025*qbin, 'alpha': alphabin + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho}
params_bininit['alpha'] = np.pi - np.radians(params_bininit['alpha']) #alpha RT Model/vbml in radians
with open("/Users/murlidhar.4/Documents/Projects/microlens/Multiple_lc_run2-Me/RTtest_event1/initcond.txt", "w") as f:
    f.write('%.3f %.4f %.3f %.3f %.4f %.2f %.2f'%(params_bininit['s'], params_bininit['q'], params_bininit['u_0'], params_bininit['alpha'], params_bininit['rho'], params_bininit['t_E'], params_bininit['t_0']))


rtm = rt.RTModel('/Users/murlidhar.4/Documents/Projects/microlens/Multiple_lc_run2-Me/RTtest_event1')
#rtm.config_Reader(binning = 1000000, renormalize = 0, thresholdoutliers = 1000000)

rtm.Reader()
#rtm.InitCond()
rtm.config_LevMar(nfits=20, timelimit=1200.0, maxsteps = 1000, bumperpower=3.5)
#rtm.launch_fits('LS')
rtm.LevMar('LS01', parameters_file = '/Users/murlidhar.4/Documents/Projects/microlens/Multiple_lc_run2-Me/RTtest_event1/initcond.txt')