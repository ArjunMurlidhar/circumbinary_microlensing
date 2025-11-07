import os
import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm
import scipy.optimize as op
import astropy.units as u
from scipy.optimize import Bounds
import matplotlib.lines as mlines
from iminuit import Minuit
from iminuit import minimize as iminuit_minimize
from scipy.optimize import OptimizeResult
import time

def magtoflux(mag, zeropint = 27.615):
    return 10.**(-0.4*(mag - zeropint))

def chi2_for_model(theta, event, parameters_to_fit, data_ref):
    """
    for given event set attributes from parameters_to_fit
    (list of str) to values from the theta list
    """
    for (index, key) in enumerate(parameters_to_fit):
        if (key == 't_E' or key =='rho') and theta[index] < 0.:
            return np.inf
        setattr(event.model.parameters, key, theta[index])
    return event.get_chi2_for_dataset(data_ref)

def fit_model(event, parameters_to_fit, bounds = None, data_ref = 0):
    """
    Fit an "event" with "parameters_to_fit" as free parameters.
    
    event = a MulensModel event
    parameters_to_fit = list of parameters to fit
    """
    # Take the initial starting point from the event.
    x0 = []
    for key in parameters_to_fit:
        value = getattr(event.model.parameters, key)
        if isinstance(value, u.Quantity):
            x0.append(value.value)
        else:
            x0.append(value)

    # *Execute fit using a 'Nelder-Mead' algorithm*
    start_time = time.time()
    result = op.minimize(
       chi2_for_model, x0, args=(event, parameters_to_fit, data_ref), bounds=bounds,
       method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': False, 'maxfev' : 600}) #10^-14, 5000
    end_time = time.time()
    print("Time taken for fit: ", end_time - start_time)
    #result = op.minimize(chi2_for_model, x0, args=(event, parameters_to_fit), method= 'L-BFGS-B' , options={'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 2000}, bounds=bounds)

    return result

def find_residuals(params, event, parameters_to_fit):
    for (index, key) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, key, params[key].value)
    event.fit_fluxes()
    srcflux, bldflux = event.get_ref_fluxes()
    modelmag = event.model.get_lc(event.datasets[0].time, source_flux=srcflux, blend_flux=bldflux)
    modelflux = mm.Utils.get_flux_from_mag(modelmag)
    residuals = (event.datasets[0].flux - modelflux) / event.datasets[0].err_flux
    return residuals  

def fit_model_minuit(event, parameters_to_fit, bounds = None, data_ref = 0):
    """
    Fit an "event" with "parameters_to_fit" as free parameters.
    
    event = a MulensModel event
    parameters_to_fit = list of parameters to fit
    """
    # Take the initial starting point from the event.
    x0 = []
    for key in parameters_to_fit:
        value = getattr(event.model.parameters, key)
        if isinstance(value, u.Quantity):
            x0.append(value.value)
        else:
            x0.append(value)
    
    def wrapped(func, args):
        return lambda x: func(x, *args)
    
    chisq_wrapped = wrapped(chi2_for_model, (event, parameters_to_fit, data_ref))
    chisq_wrapped.errordef = 1.0
    m = Minuit(chisq_wrapped, x0)
    m.limits = bounds
    m.tol = 0.0001 #0.0001
    m.strategy = 1
    start_time = time.time()
    print("Start simplex")
    m.simplex(ncall=300) #5000
    print("End Simplex")
    m.strategy = 1
    m.tol = 0.1 #0.0001
    print("Start Migrad")
    m.migrad(ncall = 300) #5000
    end_time = time.time()
    print("Time taken for fit: ", end_time - start_time)
    if m.valid:
        message = "Optimization terminated successfully"
        if m.accurate:
            message += "."
        else:
            message += ", but uncertainties are unrealiable."
    else:
        message = "Optimization failed."
        fmin = m.fmin
        if fmin.has_reached_call_limit:
            message += " Call limit was reached."
        if fmin.is_above_max_edm:
            message += " Estimated distance to minimum too large."

    n = len(x0)
    result = OptimizeResult(
        x=np.array(m.values),
        success=m.valid,
        fun=m.fval,
        hess_inv=m.covariance if m.covariance is not None else np.ones((n, n)),
        message=message,
        nfev=m.nfcn,
        njev=m.ngrad,
        minuit=m,
    )

    return result
    

#falphal = np.linspace(0.0, np.pi, 6, endpoint=False)
#fu0l = [ -0.2, -0.1, -0.075, -0.050, -0.025, -0.015, 0.0, 0.015, 0.025, 0.05, 0.075, 0.1, 0.2]
# Get input as a string separated by spaces
#input_u0 = input("Enter u_0 values separated by spaces: ")

# Split the string into a list of strings
#input_list = input_u0.split()

# Convert to integers if needed
fu0l = [-0.1, -0.075]#[float(x) for x in input_list]

#input_alpha = input("Enter start, end and number of values for alpha separated by spaces: ")
 
#input_list = input_alpha.split()

falphal = [0.0]#np.linspace(float(input_list[0]), float(input_list[1]), int(input_list[2]), endpoint=False)

filenamein = "../../"#input("Enter the filepath to Multiple_lc_run1: ")

for falpha in falphal:
    for fu0 in fu0l:
        #if (falpha == 0.0 and fu0 == -0.2) or (falpha == 0.0 and fu0 == -0.1):
            #continue
        start_time = time.time()
        alpha = 360.0 - falpha
        file_name = filenamein+"Multiple_lc_run1/s20p2s31p0psi60.0alpha%.1f/u%.3falpha%d"%(falpha,fu0,alpha)
        print("Working on u0 = %.3f and alpha = %.1f"%(fu0,alpha))
        #Read CB data
        hjd, cbflux, cbflux_err = np.loadtxt(file_name+"_flux.txt", usecols=[0,1,2], unpack=True)
        my_data = mm.MulensData([hjd, cbflux, cbflux_err], phot_fmt="flux", chi2_fmt="flux")
        print("{:} file was imported".format(file_name))
        true_params = np.loadtxt(file_name+"_info.txt", skiprows=1)
        params_cb = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s2": true_params[3], "q2": true_params[4], "alpha": true_params[5], "rho": true_params[6], "s3": true_params[7], "q3": true_params[8], "psi": true_params[9]}

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

        true_chisq = len(cbflux) - len(params_bin)
        true_chisq_pspl = len(cbflux) - 4




        #Create models
        t_0 = params_bin['t_0']
        u_0 = params_bin['u_0']
        t_E = params_bin['t_E']
        rho = params_bin['rho']
        sbin = params_bin['s']
        qbin = params_bin['q']
        alphabin = params_bin['alpha']
        splanet = params_planet['s']
        qplanet = params_planet['q']
        alphaplanet = params_planet['alpha']
        u0planet = params_planet['u_0']
        gamma = 1.25

        pspl_model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'rho': rho})
        pspl_model.set_magnification_methods([t_0 - gamma*t_E, 'finite_source_uniform_Gould94', t_0 + gamma*t_E])
        planet_model = mm.Model({'t_0': t_0, 'u_0': u0planet, 't_E': t_E, 's': splanet, 'q': qplanet, 'alpha': alphaplanet, 'rho': rho})
        planet_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
        binary_model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': sbin, 'q': qbin, 'alpha': alphabin, 'rho': rho})
        binary_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])



        #Fit PSPL model

        my_event = mm.Event(datasets=[my_data], model=pspl_model)
        init_fit = mm.FitData(dataset=my_data, model=pspl_model)
        init_fit.fit_fluxes()
        initsourceflux = init_fit.source_flux
        initblendflux = init_fit.blend_flux
        deltax2init = my_event.get_chi2_for_dataset(0)- true_chisq_pspl
        print("Initial delta chi-square PSPL = ", deltax2init)

        parameters_to_fit = ["t_0", "u_0", "t_E", "rho"]
        bounds = ((t_0 - 100, t_0 + 100), (-10.0, 10.0), (10., 100.), (0.0, 1.0)) 
        result = fit_model_minuit(my_event, parameters_to_fit, bounds)
        print("Fitting was successful? {:}".format(result.success))
        if not result.success:
            print(result.message)
        print("Function evaluations: {:}".format(result.nfev))
        if isinstance(result.fun, np.ndarray):
            if result.fun.ndim == 0:
                result_fun = float(result.fun)
            else:
                result_fun = result.fun[0]
        else:
            result_fun = result.fun
        print("The smallest function value: {:.3f}".format(result_fun))
        print("for parameters: {:.5f} {:.4f} {:.3f} {:.3f}".format(*result.x.tolist()))
        finsourceflux = my_event.get_flux_for_dataset(0)[0]
        finblendflux = my_event.get_flux_for_dataset(0)[1]
        deltax2fin = my_event.get_chi2_for_dataset(0)- true_chisq
        pspldeltax2 = deltax2fin
        print("Final delta chi-square PSPL = ", deltax2fin)
        psplmagfin = my_event.model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
        psplfluxfin = mm.Utils.get_flux_from_mag(psplmagfin)
        params_psplfinal = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3]}
        path = filenamein+"Multiple_lc_run1_fits/s20p2s31p0psi60.0alpha%.1f/"%(falpha)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
            f.write("# t_0     u_0      t_E        rho      s         q         alpha       source_flux     blend_flux     delta_chisq     flag\n")
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-PSPL\n"%(t_0, u_0, t_E, rho, 0.0, 0.0, 0.0, initsourceflux, initblendflux, deltax2init))
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Final-PSPL\n"%(float(params_psplfinal["t_0"]), float(params_psplfinal["u_0"]), float(params_psplfinal["t_E"]), float(params_psplfinal["rho"]), 0.0, 0.0, 0.0, finsourceflux[0], finblendflux, deltax2fin))
        if deltax2fin < 160.0:
            with open(filenamein+"Multiple_lc_run1_fits/not_cb.txt", "a") as f:
                f.write("{:} {:}  PSPL\n".format(falpha, fu0))
            #Plot data with PSPL model
            #Data
            fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
            dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(params_cb['t_0'], params_cb['u_0'], params_cb['t_E'], params_cb['s2'], params_cb['q2'], params_cb['alpha'], params_cb['rho'], params_cb['s3'], params_cb['q3'], params_cb['psi']))
            ax.set_xlabel('HJD')
            ax.set_ylabel('log(Flux)')
            ax.set_title('Data and binary lens models')
            plt.legend(loc='best')
            #PSPL
            ax.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label= r'PSPL Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f'%(params_psplfinal['t_0'], params_psplfinal['u_0'], params_psplfinal['t_E'], params_psplfinal['rho']))
            black_line = mlines.Line2D([], [], color='black', linestyle=':', label='PSPL Delta Chi2 = %.1f'%(pspldeltax2))
            handles, labels = ax.get_legend_handles_labels()
            legend1 = ax.legend(handles, labels, loc='lower center')
            ax.add_artist(legend1)
            legend2 = ax.legend(handles = [black_line], loc='upper left')
            ax.add_artist(legend2)
            #Inset axes
            x1, x2, y1, y2 = 1993, 2007, np.log10(np.max(my_data.flux)) - 0.8, np.log10(np.max(my_data.flux))  # subregion of the original image
            axins = ax.inset_axes(
                [0.64,0.64,0.35,0.33],
                xlim=(x1, x2), ylim=(y1, y2), xticks=[])
            axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)
            axins.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5)
              #residuals
            residualspspl = (my_data.flux - psplfluxfin)/my_data.flux
            axins2 = ax.inset_axes(
                [0.64,0.48,0.35,0.14],
                xlim=(x1, x2))
            axins2.scatter(my_data.time, residualspspl , color='black', s=0.5, label='PSPL')
            axins2.set_xlabel('HJD')
            axins2.set_ylabel('Residuals')
            plt.subplots_adjust(hspace=0.1)
            #Save figure
            plt.savefig(path+"u%.3falpha%d_plot.png"%(fu0,alpha))
            plt.close()
            continue





   #Fit planetary LC with planet model
        flagplanet = 0
        sgns = np.random.choice([-1,1], size = 7)
        planet2_model = mm.Model({'t_0': t_0 + sgns[0]*0.25, 'u_0': u0planet + sgns[1]*0.025*u0planet, 't_E': t_E + sgns[2]*0.025*t_E, 's': splanet + sgns[3]*0.025*splanet, 'q': qplanet + sgns[4]*0.025*qplanet, 'alpha': alphaplanet + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho})
        planet2_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
        params_planetinit = {'t_0': t_0 + sgns[0]*0.25, 'u_0': u0planet + sgns[1]*0.025*u0planet, 't_E': t_E + sgns[2]*0.025*t_E, 's': splanet + sgns[3]*0.025*splanet, 'q': qplanet + sgns[4]*0.025*qplanet, 'alpha': alphaplanet + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho}
        my_event_planet = mm.Event(datasets=[my_data_planet], model=planet2_model)
        init_fit = mm.FitData(dataset=my_data_planet, model=planet2_model)
        init_fit.fit_fluxes()
        initsourceflux = init_fit.source_flux
        initblendflux = init_fit.blend_flux
        deltax2init = my_event_planet.get_chi2_for_dataset(0)- true_chisq
        print("Initial delta chi-square planet-planet = ", deltax2init)
        parameters_to_fit = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
        bounds = ((t_0 - 100, t_0 + 100), (-10.0, 10.0), (10., 100.), (0.0, 1.0), (0.001, 10.0),(0.0, 1.0), (0.0, 360.)) 
        try:
            result = fit_model_minuit(my_event_planet, parameters_to_fit, bounds)
        except MemoryError:
            with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                f.write("{:} {:}  Planet-planet fit failed\n".format(falpha, fu0))
            continue

        print("Fitting was successful? {:}".format(result.success))
        if not result.success:
            print(result.message)
        print("Function evaluations: {:}".format(result.nfev))
        if isinstance(result.fun, np.ndarray):
            if result.fun.ndim == 0:
                result_fun = float(result.fun)
            else:
                result_fun = result.fun[0]
        else:
            result_fun = result.fun
        print("The smallest function value: {:.3f}".format(result_fun))#result.chisqr
        print("for parameters: , {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))#result.params)#
        finsourceflux = my_event_planet.get_flux_for_dataset(0)[0]
        finblendflux = my_event_planet.get_flux_for_dataset(0)[1]
        deltax2fin = my_event_planet.get_chi2_for_dataset(0)- true_chisq
        print("Final delta chi-square planet-planet = ", deltax2fin)
        params_planet2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 

        if deltax2fin > 160.0: #try fitting with nelder mead again
            print("Trying to fit with Nelder-Mead again")
            sgns = np.random.choice([-1,1], size = 7)
            planet2_model = mm.Model({'t_0': t_0 + sgns[0]*0.25, 'u_0': u0planet + sgns[1]*0.025*u0planet, 't_E': t_E + sgns[2]*0.025*t_E, 's': splanet + sgns[3]*0.025*splanet, 'q': qplanet + sgns[4]*0.025*qplanet, 'alpha': alphaplanet + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho})
            planet2_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
            params_planetinit = {'t_0': t_0 + sgns[0]*0.25, 'u_0': u0planet + sgns[1]*0.025*u0planet, 't_E': t_E + sgns[2]*0.025*t_E, 's': splanet + sgns[3]*0.025*splanet, 'q': qplanet + sgns[4]*0.025*qplanet, 'alpha': alphaplanet + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho}
            my_event_planet = mm.Event(datasets=[my_data_planet], model=planet2_model)
            init_fit = mm.FitData(dataset=my_data_planet, model=planet2_model)
            init_fit.fit_fluxes()
            initsourceflux = init_fit.source_flux
            initblendflux = init_fit.blend_flux
            deltax2init = my_event_planet.get_chi2_for_dataset(0)- true_chisq
            print("Initial delta chi-square planet-planet = ", deltax2init)
            try:
                result = fit_model(my_event_planet, parameters_to_fit, bounds)
            except MemoryError:
                with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                    f.write("{:} {:}  Planet-planet fit failed\n".format(falpha, fu0))
                continue
            print("Fitting was successful? {:}".format(result.success))
            if not result.success:
                print(result.message)
            print("Function evaluations: {:}".format(result.nfev))
            if isinstance(result.fun, np.ndarray):
                if result.fun.ndim == 0:
                    result_fun = float(result.fun)
                else:
                    result_fun = result.fun[0]
            else:
                result_fun = result.fun
            print("The smallest function value: {:.3f}".format(result_fun))#result.chisqr
            print("for parameters: , {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))#result.params)#
            finsourceflux = my_event_planet.get_flux_for_dataset(0)[0]
            finblendflux = my_event_planet.get_flux_for_dataset(0)[1]
            deltax2fin2 = my_event_planet.get_chi2_for_dataset(0)- true_chisq
            if deltax2fin2 < 160.0:
                flagplanet = 1
                print("Final delta chi-square planet-planet = ", deltax2fin2)
                params_planet2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 
                with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                    f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Planet only\n"%(params_planetinit["t_0"],params_planetinit["u_0"], params_planetinit["t_E"], params_planetinit["rho"], params_planetinit["s"], params_planetinit["q"], params_planetinit["alpha"], initsourceflux, initblendflux, deltax2init))
                    f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Planet only\n"%(params_planet2final["t_0"], params_planet2final["u_0"], params_planet2final["t_E"], params_planet2final["rho"], params_planet2final["s"], params_planet2final["q"], params_planet2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
            elif deltax2fin2 > 160.0:
                if deltax2fin < deltax2fin2:
                    print("Final delta chi-square planet-planet = ", deltax2fin)
                    with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Planet only\n"%(params_planetinit["t_0"],params_planetinit["u_0"], params_planetinit["t_E"], params_planetinit["rho"], params_planetinit["s"], params_planetinit["q"], params_planetinit["alpha"], initsourceflux, initblendflux, deltax2init))
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Planet only\n"%(params_planet2final["t_0"], params_planet2final["u_0"], params_planet2final["t_E"], params_planet2final["rho"], params_planet2final["s"], params_planet2final["q"], params_planet2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
                else:
                    flagplanet = 1
                    print("Final delta chi-square planet-planet = ", deltax2fin2)
                    params_planet2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 
                    with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Planet only\n"%(params_planetinit["t_0"],params_planetinit["u_0"], params_planetinit["t_E"], params_planetinit["rho"], params_planetinit["s"], params_planetinit["q"], params_planetinit["alpha"], initsourceflux, initblendflux, deltax2init))
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Planet only\n"%(params_planet2final["t_0"], params_planet2final["u_0"], params_planet2final["t_E"], params_planet2final["rho"], params_planet2final["s"], params_planet2final["q"], params_planet2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))

                with open(filenamein+"Multiple_lc_run1_fits/Binaryfit_fail.txt", "a") as f:
                    f.write("{:} {:}  Planet failed\n".format(falpha, fu0))

        else:
            with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Planet only\n"%(params_planetinit["t_0"],params_planetinit["u_0"], params_planetinit["t_E"], params_planetinit["rho"], params_planetinit["s"], params_planetinit["q"], params_planetinit["alpha"], initsourceflux, initblendflux, deltax2init))
                f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Planet only\n"%(params_planet2final["t_0"], params_planet2final["u_0"], params_planet2final["t_E"], params_planet2final["rho"], params_planet2final["s"], params_planet2final["q"], params_planet2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))

            






        #Fit a planet model
        my_event.model = planet_model
        init_fit = mm.FitData(dataset=my_data, model=planet_model)
        init_fit.fit_fluxes()
        initsourceflux = init_fit.source_flux
        initblendflux = init_fit.blend_flux
        planetmaginit = planet_model.get_lc(my_data.time, source_flux=initsourceflux, blend_flux=initblendflux)
        planetfluxinit = mm.Utils.get_flux_from_mag(planetmaginit)
        deltax2init = my_event.get_chi2_for_dataset(0)- true_chisq
        print("Initial delta chi-square planet = ", deltax2init)

        parameters_to_fit = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
        bounds = ((t_0 - 100, t_0 + 100), (-10.0, 10.0), (10., 100.), (0.0, 1.0), (0.001, 10.0),(0.0, 1.0), (0.0, 360.)) 
        try:
            if flagplanet == 0:
                result = fit_model_minuit(my_event, parameters_to_fit, bounds)
            else:
                result = fit_model(my_event, parameters_to_fit, bounds)
        except MemoryError:
            with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                f.write("{:} {:}  Planet fit failed\n".format(falpha, fu0))
            continue
        print("Fitting was successful? {:}".format(result.success))
        if not result.success:
            print(result.message)
        print("Function evaluations: {:}".format(result.nfev))
        if isinstance(result.fun, np.ndarray):
            if result.fun.ndim == 0:
                result_fun = float(result.fun)
            else:
                result_fun = result.fun[0]
        else:
            result_fun = result.fun
        print("The smallest function value: {:.3f}".format(result_fun))
        print("for parameters: {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))
        finsourceflux = my_event.get_flux_for_dataset(0)[0]
        finblendflux = my_event.get_flux_for_dataset(0)[1]
        deltax2fin = my_event.get_chi2_for_dataset(0)- true_chisq
        planetdeltax2 = deltax2fin
        print("Final delta chi-square planet = ", deltax2fin)
        planetmagfin = my_event.model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
        planetfluxfin = mm.Utils.get_flux_from_mag(planetmagfin)
        params_planetfinal = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]}
        with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Planet\n"%(t_0, u0planet, t_E, rho, splanet, qplanet, alphaplanet, initsourceflux, initblendflux, deltax2init))
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Final-Planet\n"%(params_planetfinal["t_0"], params_planetfinal["u_0"], params_planetfinal["t_E"], params_planetfinal["rho"], params_planetfinal["s"], params_planetfinal["q"], params_planetfinal["alpha"], finsourceflux[0], finblendflux, deltax2fin))
        if deltax2fin < 160.0:
            with open(filenamein+"Multiple_lc_run1_fits/not_cb.txt", "a") as f:
                f.write("{:} {:}  Planet\n".format(falpha, fu0))
            #Plot data with PSPL and planet models
             #Data
            fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
            dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(params_cb['t_0'], params_cb['u_0'], params_cb['t_E'], params_cb['s2'], params_cb['q2'], params_cb['alpha'], params_cb['rho'], params_cb['s3'], params_cb['q3'], params_cb['psi']))
            ax.set_xlabel('HJD')
            ax.set_ylabel('log(Flux)')
            ax.set_title('Data and binary lens models')
            plt.legend(loc='best')
            #PSPL
            ax.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label= r'PSPL Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f'%(params_psplfinal['t_0'], params_psplfinal['u_0'], params_psplfinal['t_E'], params_psplfinal['rho']))
            black_line = mlines.Line2D([], [], color='black', linestyle=':', label='PSPL Delta Chi2 = %.1f'%(pspldeltax2))
            #Planet
            ax.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label= r'Planet Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_planetfinal['t_0'], params_planetfinal['u_0'], params_planetfinal['t_E'], params_planetfinal['rho'], params_planetfinal['s'], params_planetfinal['q'], params_planetfinal['alpha']))
            red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Planet Delta Chi2 = %.1f'%(planetdeltax2))
            handles, labels = ax.get_legend_handles_labels()
            legend1 = ax.legend(handles, labels, loc='lower center')
            ax.add_artist(legend1)
            legend2 = ax.legend(handles = [black_line, red_line], loc='upper left')
            ax.add_artist(legend2)
            #inset axes
            x1, x2, y1, y2 = 1993, 2007, np.log10(np.max(my_data.flux)) - 0.8, np.log10(np.max(my_data.flux))  # subregion of the original image
            axins = ax.inset_axes(
                [0.64,0.64,0.35,0.33],
                xlim=(x1, x2), ylim=(y1, y2), xticks=[])
            axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)
            axins.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label='PSPL Model')
            axins.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label='Planet Model')
            #residuals
            residualspspl = (my_data.flux - psplfluxfin)/my_data.flux
            residualsplan = (my_data.flux - planetfluxfin)/my_data.flux
            axins2 = ax.inset_axes(
                [0.64,0.48,0.35,0.14],
                xlim=(x1, x2))
            axins2.scatter(my_data.time, residualspspl , color='black', s=0.5, label='PSPL')
            axins2.scatter(my_data.time, residualsplan , color='red', s=0.5, label='Planet')
            handles, labels = axins2.get_legend_handles_labels()
            legend1 = axins2.legend(handles, labels, loc='best')
            axins2.add_artist(legend1)
            axins2.set_xlabel('HJD')
            axins2.set_ylabel('Residuals')
            plt.subplots_adjust(hspace=0.1)
            #Save figure
            plt.savefig(path+"u%.3falpha%d_plot.png"%(fu0,alpha))
            plt.close()
            continue








        #Fit a binary lc with binary model
        flagbinary = 0
        sgns = np.random.choice([-1,1], size = 7)
        binary2_model = mm.Model({'t_0': t_0 + sgns[0]*0.25, 'u_0': u_0 + sgns[1]*0.025*u_0, 't_E': t_E + sgns[2]*0.025*t_E, 's': sbin + sgns[3]*0.025*sbin, 'q': qbin + sgns[4]*0.025*qbin, 'alpha': alphabin + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho})
        binary2_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
        params_bininit = {'t_0': t_0 + sgns[0]*0.25, 'u_0': u_0 + sgns[1]*0.025*u_0, 't_E': t_E + sgns[2]*0.025*t_E, 's': sbin + sgns[3]*0.025*sbin, 'q': qbin + sgns[4]*0.025*qbin, 'alpha': alphabin + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho}
        my_event_binary = mm.Event(datasets=[my_data_bin], model=binary2_model)
        init_fit = mm.FitData(dataset=my_data_bin, model=binary2_model)
        init_fit.fit_fluxes()
        initsourceflux = init_fit.source_flux
        initblendflux = init_fit.blend_flux
        deltax2init = my_event_binary.get_chi2_for_dataset(0)- true_chisq
        print("Initial delta chi-square binary-binary= ", deltax2init)
        parameters_to_fit = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
        bounds = ((t_0 - 100, t_0 + 100), (-10.0, 10.0), (10., 100.), (0.0, 1.0), (0.001, 10.0),(0.0, 1.0), (0.0, 360.)) 
        try:
            result = fit_model_minuit(my_event_binary, parameters_to_fit, bounds)
        except MemoryError:
            with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                f.write("{:} {:}  Binary-binary fit failed\n".format(falpha, fu0))
            continue
        print("Fitting was successful? {:}".format(result.success))
        if not result.success:
            print(result.message)
        print("Function evaluations: {:}".format(result.nfev))
        if isinstance(result.fun, np.ndarray):
            if result.fun.ndim == 0:
                result_fun = float(result.fun)
            else:
                result_fun = result.fun[0]
        else:
            result_fun = result.fun
        print("The smallest function value: {:.3f}".format(result_fun))#result.chisqr
        print("for parameters: , {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))#result.params)#
        finsourceflux = my_event_binary.get_flux_for_dataset(0)[0]
        finblendflux = my_event_binary.get_flux_for_dataset(0)[1]
        deltax2fin = my_event_binary.get_chi2_for_dataset(0)- true_chisq
        print("Final delta chi-square binary-binary= ", deltax2fin)
        params_bin2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 

        if deltax2fin > 160.0: #try fitting with nelder mead again
            print("Trying to fit with Nelder-Mead again")
            sgns = np.random.choice([-1,1], size = 7)
            binary2_model = mm.Model({'t_0': t_0 + sgns[0]*0.25, 'u_0': u_0 + sgns[1]*0.025*u_0, 't_E': t_E + sgns[2]*0.025*t_E, 's': sbin + sgns[3]*0.025*sbin, 'q': qbin + sgns[4]*0.025*qbin, 'alpha': alphabin + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho})
            binary2_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
            params_bininit = {'t_0': t_0 + sgns[0]*0.25, 'u_0': u_0 + sgns[1]*0.025*u_0, 't_E': t_E + sgns[2]*0.025*t_E, 's': sbin + sgns[3]*0.025*sbin, 'q': qbin + sgns[4]*0.025*qbin, 'alpha': alphabin + sgns[5]*0.5, 'rho': rho + sgns[6]*0.025*rho}
            my_event_binary = mm.Event(datasets=[my_data_bin], model=binary2_model)
            init_fit = mm.FitData(dataset=my_data_bin, model=binary2_model)
            init_fit.fit_fluxes()
            initsourceflux = init_fit.source_flux
            initblendflux = init_fit.blend_flux
            deltax2init = my_event_binary.get_chi2_for_dataset(0)- true_chisq
            print("Initial delta chi-square binary-binary= ", deltax2init)
            try:
                result = fit_model(my_event_binary, parameters_to_fit, bounds)
            except MemoryError:
                with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                    f.write("{:} {:}  Binary-binary fit failed\n".format(falpha, fu0))
                continue
            print("Fitting was successful? {:}".format(result.success))
            if not result.success:
                print(result.message)
            print("Function evaluations: {:}".format(result.nfev))
            if isinstance(result.fun, np.ndarray):
                if result.fun.ndim == 0:
                    result_fun = float(result.fun)
                else:
                    result_fun = result.fun[0]
            else:
                result_fun = result.fun
            print("The smallest function value: {:.3f}".format(result_fun))#result.chisqr
            print("for parameters: , {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))#result.params)#
            finsourceflux = my_event_binary.get_flux_for_dataset(0)[0]
            finblendflux = my_event_binary.get_flux_for_dataset(0)[1]
            deltax2fin2 = my_event_binary.get_chi2_for_dataset(0)- true_chisq
            if deltax2fin2 < 160.0:
                flagbinary = 1
                print("Final delta chi-square binary-binary= ", deltax2fin2)
                params_bin2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 
                with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                    f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Binary only\n"%(params_bininit["t_0"],params_bininit["u_0"], params_bininit["t_E"], params_bininit["rho"], params_bininit["s"], params_bininit["q"], params_bininit["alpha"], initsourceflux, initblendflux, deltax2init))
                    f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Binary only\n"%(params_bin2final["t_0"], params_bin2final["u_0"], params_bin2final["t_E"], params_bin2final["rho"], params_bin2final["s"], params_bin2final["q"], params_bin2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
            elif deltax2fin2 > 160.0:
                if deltax2fin < deltax2fin2:
                    print("Final delta chi-square binary-binary= ", deltax2fin)
                    with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Binary only\n"%(params_bininit["t_0"],params_bininit["u_0"], params_bininit["t_E"], params_bininit["rho"], params_bininit["s"], params_bininit["q"], params_bininit["alpha"], initsourceflux, initblendflux, deltax2init))
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Binary only\n"%(params_bin2final["t_0"], params_bin2final["u_0"], params_bin2final["t_E"], params_bin2final["rho"], params_bin2final["s"], params_bin2final["q"], params_bin2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
                else:
                    flagbinary = 1
                    print("Final delta chi-square binary-binary= ", deltax2fin2)
                    params_bin2final = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]} 
                    with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Binary only\n"%(params_bininit["t_0"],params_bininit["u_0"], params_bininit["t_E"], params_bininit["rho"], params_bininit["s"], params_bininit["q"], params_bininit["alpha"], initsourceflux, initblendflux, deltax2init))
                        f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Binary only\n"%(params_bin2final["t_0"], params_bin2final["u_0"], params_bin2final["t_E"], params_bin2final["rho"], params_bin2final["s"], params_bin2final["q"], params_bin2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
                
                with open(filenamein+"Multiple_lc_run1_fits/Binaryfit_fail.txt", "a") as f:
                    f.write("{:} {:}  Binary failed\n".format(falpha, fu0))
            
        else:
            with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
                f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Binary only\n"%(params_bininit["t_0"],params_bininit["u_0"], params_bininit["t_E"], params_bininit["rho"], params_bininit["s"], params_bininit["q"], params_bininit["alpha"], initsourceflux, initblendflux, deltax2init))
                f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Binary only\n"%(params_bin2final["t_0"], params_bin2final["u_0"], params_bin2final["t_E"], params_bin2final["rho"], params_bin2final["s"], params_bin2final["q"], params_bin2final["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
                                                                                                                                     









        #Fit a binary model
        my_event.model = binary_model
        init_fit = mm.FitData(dataset=my_data, model=binary_model)
        init_fit.fit_fluxes()
        initsourceflux = init_fit.source_flux
        initblendflux = init_fit.blend_flux
        binmaginit = binary_model.get_lc(my_data.time, source_flux=initsourceflux, blend_flux=initblendflux)
        binfluxinit = mm.Utils.get_flux_from_mag(binmaginit)
        deltax2init = my_event.get_chi2_for_dataset(0)- true_chisq
        print("Initial delta chi-square binary = ", deltax2init)

        parameters_to_fit = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
        bounds = ((t_0 - 100, t_0 + 100), (-10.0, 10.0), (10., 100.), (0.0, 1.0), (0.001, 10.0),(0.0, 1.0), (0.0, 360.)) 
        try:
            if flagbinary == 0:
                result = fit_model_minuit(my_event, parameters_to_fit, bounds)
            else:
                result = fit_model(my_event, parameters_to_fit, bounds)
        except MemoryError:
            with open(filenamein+"Multiple_lc_run1_fits/Memory_error.txt", "a") as f:
                f.write("{:} {:}  Binary fit failed\n".format(falpha, fu0))
            continue
        print("Fitting was successful? {:}".format(result.success))
        if not result.success:
            print(result.message)
        print("Function evaluations: {:}".format(result.nfev))
        if isinstance(result.fun, np.ndarray):
            if result.fun.ndim == 0:
                result_fun = float(result.fun)
            else:
                result_fun = result.fun[0]
        else:
            result_fun = result.fun
        print("The smallest function value: {:.3f}".format(result_fun))
        print("for parameters: {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))
        finsourceflux = my_event.get_flux_for_dataset(0)[0]
        finblendflux = my_event.get_flux_for_dataset(0)[1]
        deltax2fin = my_event.get_chi2_for_dataset(0)- true_chisq
        binarydeltax2 = deltax2fin
        print("Final delta chi-square binary = ", deltax2fin)
        binmagfin = my_event.model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
        binfluxfin = mm.Utils.get_flux_from_mag(binmagfin)
        params_binfinal = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]}
        with open(path+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f  Initial-Binary\n"%(t_0, u_0, t_E, rho, sbin, qbin, alphabin, initsourceflux, initblendflux, deltax2init))
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.4f    %.2f    %.1f   %.1f   %.1f   %.1f  Final-Binary\n"%(params_binfinal["t_0"], params_binfinal["u_0"], params_binfinal["t_E"], params_binfinal["rho"], params_binfinal["s"], params_binfinal["q"], params_binfinal["alpha"], finsourceflux[0], finblendflux, deltax2fin,result_fun))
        if deltax2fin < 160.0:
            with open(filenamein+"Multiple_lc_run1_fits/not_cb.txt", "a") as f:
                f.write("{:} {:}  Binary\n".format(falpha, fu0))
            #Plot data with PSPL, planet and binary final models
            #Data
            fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
            dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(params_cb['t_0'], params_cb['u_0'], params_cb['t_E'], params_cb['s2'], params_cb['q2'], params_cb['alpha'], params_cb['rho'], params_cb['s3'], params_cb['q3'], params_cb['psi']))
            ax.set_xlabel('HJD')
            ax.set_ylabel('log(Flux)')
            ax.set_title('Data and binary lens models')
            plt.legend(loc='best')
            #PSPL
            ax.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label= r'PSPL Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f'%(params_psplfinal['t_0'], params_psplfinal['u_0'], params_psplfinal['t_E'], params_psplfinal['rho']))
            black_line = mlines.Line2D([], [], color='black', linestyle=':', label='PSPL Delta Chi2 = %.1f'%(pspldeltax2))
            #Planet
            ax.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label= r'Planet Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_planetfinal['t_0'], params_planetfinal['u_0'], params_planetfinal['t_E'], params_planetfinal['rho'], params_planetfinal['s'], params_planetfinal['q'], params_planetfinal['alpha']))
            red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Planet Delta Chi2 = %.1f'%(planetdeltax2))
            #Binary
            ax.plot(my_data.time, np.log10(binfluxfin), color='blue', linestyle='-', markersize=0.5, label= r'Binary Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_binfinal['t_0'], params_binfinal['u_0'], params_binfinal['t_E'], params_binfinal['rho'], params_binfinal['s'], params_binfinal['q'], params_binfinal['alpha']))
            blue_line = mlines.Line2D([], [], color='blue', linestyle='-', label='Binary Delta Chi2 = %.1f'%(binarydeltax2))
            handles, labels = ax.get_legend_handles_labels()
            legend1 = ax.legend(handles, labels, loc='lower center')
            ax.add_artist(legend1)
            legend2 = ax.legend(handles = [black_line, red_line, blue_line], loc='upper left')
            ax.add_artist(legend2)
            #inset axes
            x1, x2, y1, y2 = 1993, 2007, np.log10(np.max(my_data.flux)) - 0.8, np.log10(np.max(my_data.flux))  # subregion of the original image
            axins = ax.inset_axes(
                [0.64,0.64,0.35,0.33],
                xlim=(x1, x2), ylim=(y1, y2), xticks=[])
            axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)
            axins.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label='PSPL Model')
            axins.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label='Planet Model')
            axins.plot(my_data.time, np.log10(binfluxfin), color='blue', linestyle='-', markersize=0.5, label='Binary Model')
            #residuals
            residualsplan = (my_data.flux - planetfluxfin)/my_data.flux
            residualsbin = (my_data.flux - binfluxfin)/my_data.flux
            axins2 = ax.inset_axes(
                [0.64,0.48,0.35,0.14],
                xlim=(x1, x2))
            axins2.scatter(my_data.time, residualsplan , color='red', s=0.5, label='Planet')
            axins2.scatter(my_data.time, residualsbin, color='blue', s=0.5, label='Binary')
            handles, labels = axins2.get_legend_handles_labels()
            legend1 = axins2.legend(handles, labels, loc='best')
            axins2.add_artist(legend1)
            axins2.set_xlabel('HJD')
            axins2.set_ylabel('Residuals')
            plt.subplots_adjust(hspace=0.1)
            #Save figure
            plt.savefig(path+"u%.3falpha%d_plot.png"%(fu0,alpha))
            plt.close()
            continue




            

        
        #Plot data with PSPL, planet and binary final models
        #Data
        fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
        dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(params_cb['t_0'], params_cb['u_0'], params_cb['t_E'], params_cb['s2'], params_cb['q2'], params_cb['alpha'], params_cb['rho'], params_cb['s3'], params_cb['q3'], params_cb['psi']))
        ax.set_xlabel('HJD')
        ax.set_ylabel('log(Flux)')
        ax.set_title('Data and binary lens models')
        plt.legend(loc='best')
        #PSPL
        ax.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5, label= r'PSPL Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f'%(params_psplfinal['t_0'], params_psplfinal['u_0'], params_psplfinal['t_E'], params_psplfinal['rho']))
        black_line = mlines.Line2D([], [], color='black', linestyle=':', label='PSPL Delta Chi2 = %.1f'%(pspldeltax2))
        #Planet
        ax.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label= r'Planet Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_planetfinal['t_0'], params_planetfinal['u_0'], params_planetfinal['t_E'], params_planetfinal['rho'], params_planetfinal['s'], params_planetfinal['q'], params_planetfinal['alpha']))
        red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Planet Delta Chi2 = %.1f'%(planetdeltax2))
        #Binary
        ax.plot(my_data.time, np.log10(binfluxfin), color='blue', linestyle='-', markersize=0.5, label= r'Binary Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_binfinal['t_0'], params_binfinal['u_0'], params_binfinal['t_E'], params_binfinal['rho'], params_binfinal['s'], params_binfinal['q'], params_binfinal['alpha']))
        blue_line = mlines.Line2D([], [], color='blue', linestyle='-', label='Binary Delta Chi2 = %.1f'%(binarydeltax2))
        handles, labels = ax.get_legend_handles_labels()
        legend1 = ax.legend(handles, labels, loc='lower center')
        ax.add_artist(legend1)
        legend2 = ax.legend(handles = [black_line, red_line, blue_line], loc='upper left')
        ax.add_artist(legend2)
        #inset axes
        x1, x2, y1, y2 = 1993, 2007, np.log10(np.max(my_data.flux)) - 0.8, np.log10(np.max(my_data.flux))  # subregion of the original image
        axins = ax.inset_axes(
            [0.64,0.64,0.35,0.33],
            xlim=(x1, x2), ylim=(y1, y2), xticks=[])
        axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)
        axins.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5)
        axins.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5)
        axins.plot(my_data.time, np.log10(binfluxfin), color='blue', linestyle='-', markersize=0.5)
        #residuals
        residualsplan = (my_data.flux - planetfluxfin)/my_data.flux
        residualsbin = (my_data.flux - binfluxfin)/my_data.flux
        axins2 = ax.inset_axes(
            [0.64,0.48,0.35,0.14],
            xlim=(x1, x2))
        axins2.scatter(my_data.time, residualsplan , color='red', s=0.5, label='Planet')
        axins2.scatter(my_data.time, residualsbin, color='blue', s=0.5, label='Binary')
        handles, labels = axins2.get_legend_handles_labels()
        legend1 = axins2.legend(handles, labels, loc='best')
        axins2.add_artist(legend1)
        axins2.set_xlabel('HJD')
        axins2.set_ylabel('Residuals')
        plt.subplots_adjust(hspace=0.1)
        #Save figure
        plt.savefig(path+"u%.3falpha%d_plot.png"%(fu0,alpha))
        plt.close()
        end_time = time.time()
        print("Total time taken = %.2f"%(end_time - start_time))