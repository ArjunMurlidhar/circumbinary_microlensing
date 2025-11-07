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
import multiprocessing
from Plot_binary_models import plot_init_final

def magtoflux(mag, zeropint = 27.615):
    return 10.**(-0.4*(mag - zeropint))

def chi2_for_model(theta, event, parameters_to_fit, data_ref):
    """
    for given event set attributes from parameters_to_fit
    (list of str) to values from the theta list
    """
    for (index, key) in enumerate(parameters_to_fit):
        #if (key == 's' or key =='rho' or key == 'q'):
         #   theta[index] = 10.**theta[index]
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
        #if key == 's' or key == 'rho' or key == 'q':
         #   value = np.log10(value)
        if isinstance(value, u.Quantity):
            x0.append(value.value)
        else:
            x0.append(value)

    # *Execute fit using a 'Nelder-Mead' algorithm*
    result = op.minimize(
       chi2_for_model, x0, args=(event, parameters_to_fit, data_ref), bounds=bounds,
       method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': False, 'maxfev' : 1000}) #10^-14, 5000
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
        #if key == 's' or key == 'rho' or key == 'q':
            #value = np.log10(value)
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
    m.simplex(ncall=500) #5000
    m.strategy = 1
    m.tol = 0.1 #0.0001
    m.migrad(ncall = 500) #5000
    end_time = time.time()
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



def fit_planet(my_data, params_bin, true_chisq, true_chisq_pspl, filenamein, falpha, fu0, alpha, lockfit, lockbinaryfail, locknotcb, lockmem, q):    
    t_0 = params_bin['t_0']
    u_0 = params_bin['u_0']
    t_E = params_bin['t_E']
    rho = params_bin['rho']
    alphamulens = alpha 
    splanet=0.8605 
    qplanet=0.0000052 
    planetorient=67.8*np.pi/180. 
    alphaplanet = alphamulens - planetorient*180./np.pi
    u0planet = params_bin['u_0']
    gamma = 1.25

    pspl_model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'rho': rho})
    pspl_model.set_magnification_methods([t_0 - gamma*t_E, 'finite_source_uniform_Gould94', t_0 + gamma*t_E])
    planet_model = mm.Model({'t_0': t_0, 'u_0': u0planet, 't_E': t_E, 's': splanet, 'q': qplanet, 'alpha': alphaplanet, 'rho': rho})
    planet_model.set_magnification_methods([t_0 - gamma*t_E, 'VBBL', t_0 + gamma*t_E])
    #planet_model.set_magnification_methods_parameters({'VBBL': {'accuracy': 1e-4}})

    outpath = '../data_files/'
        
    #Fit a planet model
    print("Fitting planet model u_0 %.3f"%(u_0))
    start_time = time.time()
    my_event = mm.Event(datasets=[my_data], model=planet_model)
    init_fit = mm.FitData(dataset=my_data, model=planet_model)
    init_fit.fit_fluxes()
    initsourceflux = init_fit.source_flux
    initblendflux = init_fit.blend_flux
    planetmaginit = planet_model.get_lc(my_data.time, source_flux=initsourceflux, blend_flux=initblendflux)
    planetfluxinit = mm.Utils.get_flux_from_mag(planetmaginit)
    deltax2init = np.abs(my_event.get_chi2_for_dataset(0)- true_chisq)
    #print("Initial delta chi-square planet = ", deltax2init)

    parameters_to_fit = ["t_0", "u_0", "t_E", "rho", "s", "q", "alpha"]
    bounds = ((t_0 - 5., t_0 + 5.), (-10.0, 10.0), (t_E - 3., t_E + 3.), (1e-5, 1.0), (1e-2, 10.0),(1e-7, 1.0), (0.0, 360.)) 
    result = fit_model(my_event, parameters_to_fit, bounds)
    if isinstance(result.fun, np.ndarray):
        if result.fun.ndim == 0:
            result_fun = float(result.fun)
        else:
            result_fun = result.fun[0]
    else:
        result_fun = result.fun
    #print("The smallest function value: {:.3f}".format(result_fun))
    #print("for parameters: {:.5f} {:.4f} {:.3f} {:.3f} {:.3f} {:.4f} {:.2f}".format(*result.x.tolist()))
    fin_fit = mm.FitData(dataset=my_data, model=my_event.model)
    fin_fit.fit_fluxes()
    finsourceflux = fin_fit.source_flux 
    finblendflux = fin_fit.blend_flux
    #finsourceflux = my_event.get_flux_for_dataset(0)[0]
    #finblendflux = my_event.get_flux_for_dataset(0)[1]
    deltax2fin = np.abs(my_event.get_chi2_for_dataset(0)- true_chisq)
    planetdeltax2 = deltax2fin
    #print("Final delta chi-square planet = ", deltax2fin)
    planetmagfin = my_event.model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
    planetfluxfin = mm.Utils.get_flux_from_mag(planetmagfin)
    params_planetfinal = {"t_0": result.x[0], "u_0": result.x[1], "t_E": result.x[2], "rho": result.x[3], "s": result.x[4], "q": result.x[5], "alpha": result.x[6]}
    with lockfit:
        with open(outpath+"u%.3falpha%d.txt"%(fu0,alpha), "a") as f:
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.5f    %.2f    %.1f   %.1f   %.1f  Initial-Planet\n"%(t_0, u0planet, t_E, rho, splanet, qplanet, alphaplanet, initsourceflux, initblendflux, deltax2init))
            f.write(" %.1f   %.4f    %.2f   %.5f     %.3f    %.5f    %.2f    %.1f   %.1f   %.1f  Final-Planet\n"%(params_planetfinal["t_0"], params_planetfinal["u_0"], params_planetfinal["t_E"], params_planetfinal["rho"], params_planetfinal["s"], params_planetfinal["q"], params_planetfinal["alpha"], finsourceflux, finblendflux, deltax2fin))#finsourceflux[0]
    end_time = time.time()
    print("Time taken for planet fit u0 %.3f = %.2f"%(u_0, (end_time - start_time)))
    return [0, planetfluxfin, planetdeltax2, params_planetfinal]



def task_manager(filenamein, falphal, fu0l, lockbinaryfail, locknotcb, lockmem):
    print("started process for ", fu0l)
    for falpha in falphal:
        for fu0 in fu0l:
            start_time = time.time()
            outpath = "../data_files/"
            alpha = 360.0 - falpha
            inpath = "../data_files/u%.3falpha%d"%(fu0,alpha)
            print("Working on u0 = %.3f and alpha = %.1f"%(fu0,alpha))
            #Read CB data
            hjd, cbflux, cbflux_err = np.loadtxt(inpath+"_flux.txt", usecols=[0,1,2], unpack=True)
            my_data = mm.MulensData([hjd, cbflux, cbflux_err], phot_fmt="flux", chi2_fmt="flux")
            true_params = np.loadtxt(inpath+"_info.txt", skiprows=1)
            params_cb = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s2": true_params[3], "q2": true_params[4], "alpha": true_params[5], "rho": true_params[6], "s3": true_params[7], "q3": true_params[8], "psi": true_params[9]}

            true_chisq = 0. #No Gaussian scatter #len(cbflux) - len(params_bin) 
            true_chisq_pspl = 0. #No Gaussian scatter #len(cbflux) - 4


            #Fit a PSPL Model
            t_0 = params_cb['t_0']
            u_0 = params_cb['u_0']
            t_E = params_cb['t_E']
            rho = params_cb['rho']
            gamma = 1.25

            pspl_model = mm.Model({'t_0': t_0, 'u_0': u_0, 't_E': t_E, 'rho': rho})
            pspl_model.set_magnification_methods([t_0 - gamma*t_E, 'finite_source_uniform_Gould94', t_0 + gamma*t_E])

            my_event = mm.Event(datasets=[my_data], model=pspl_model)
            init_fit = mm.FitData(dataset=my_data, model=pspl_model)
            init_fit.fit_fluxes()
            initsourceflux = init_fit.source_flux
            initblendflux = init_fit.blend_flux
            deltax2init = np.abs(my_event.get_chi2_for_dataset(0)- true_chisq_pspl)
            #print("Initial delta chi-square PSPL = ", deltax2init)

            parameters_to_fit = ["t_0", "u_0", "t_E", "rho"]
            bounds = ((t_0 - 5, t_0 + 5), (-10.0, 10.0), (t_E - 3., t_E + 3.), (1e-5, 1.0)) 
            result = fit_model(my_event, parameters_to_fit, bounds)
            #print("Fitting was successful? {:}".format(result.success))
            if not result.success:
                print(result.message)
            #print("Function evaluations: {:}".format(result.nfev))
            if isinstance(result.fun, np.ndarray):
                if result.fun.ndim == 0:
                    result_fun = float(result.fun)
                else:
                    result_fun = result.fun[0]
            else:
                result_fun = result.fun
            #print("The smallest function value: {:.3f}".format(result_fun))
            #print("for parameters: {:.5f} {:.4f} {:.3f} {:.3f}".format(*result.x.tolist()))
            fin_fit = mm.FitData(dataset=my_data, model=my_event.model)
            fin_fit.fit_fluxes()
            finsourceflux = fin_fit.source_flux 
            finblendflux = fin_fit.blend_flux
            #finsourceflux = my_event.get_flux_for_dataset(0)[0]
            #   finblendflux = my_event.get_flux_for_dataset(0)[1]
            deltax2fin = np.abs(my_event.get_chi2_for_dataset(0)- true_chisq)
            pspldeltax2 = deltax2fin
            #print("Final delta chi-square PSPL = ", deltax2fin)
            psplmagfin = my_event.model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
            psplfluxfin = mm.Utils.get_flux_from_mag(psplmagfin)

    
            #Fit planet and binary models

            q = multiprocessing.Queue()
            lockfit = multiprocessing.Lock()

            result = fit_planet(my_data, params_cb, true_chisq, true_chisq_pspl, filenamein, falpha, fu0, alpha, lockfit, lockbinaryfail, locknotcb, lockmem, q)


            
            planetfluxfin = result[1]
            planetdeltax2 = result[2]
            params_planetfinal = result[3]
              
           
            
            #Plot data with PSPL, planet and binary final models
            #Data
            fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
            dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(params_cb['t_0'], params_cb['u_0'], params_cb['t_E'], params_cb['s2'], params_cb['q2'], params_cb['alpha'], params_cb['rho'], params_cb['s3'], params_cb['q3'], params_cb['psi']))
            ax.set_xlabel('HJD')
            ax.set_ylabel('log(Flux)')
            ax.set_title('Data and binary lens models')
            plt.legend(loc='best')

    
            handles2 = []
            labels2 = []
           
            #Planet
            ax.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5, label= r'Planet Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(params_planetfinal['t_0'], params_planetfinal['u_0'], params_planetfinal['t_E'], params_planetfinal['rho'], params_planetfinal['s'], params_planetfinal['q'], params_planetfinal['alpha']))
            red_line = mlines.Line2D([], [], color='red', linestyle='--', label='Planet Delta Chi2 = %.1f'%(planetdeltax2)) 
            handles2.append(red_line)
            labels2.append(red_line.get_label())
        

            handles, labels = ax.get_legend_handles_labels()
            legend1 = ax.legend(handles, labels, loc='lower center')
            ax.add_artist(legend1)
            legend2 = ax.legend(handles2, labels2, loc='upper left')
            ax.add_artist(legend2)

            residualpspl = np.abs(my_data.flux - psplfluxfin)/my_data.flux
            tsmaxdev = my_data.time[np.argmax(residualpspl)]

            #inset axes
            if np.max(residualpspl) < 0.1:
                x1, x2, y1, y2 = 2000. - 3, 2000. + 3, np.log10(np.max(my_data.flux)) - 0.5, np.log10(np.max(my_data.flux)) + 0.5
            else:
                x1, x2, y1, y2 = tsmaxdev - 3, tsmaxdev + 3, np.log10(my_data.flux[np.argmax(residualpspl)]) - 0.5, np.log10(my_data.flux[np.argmax(residualpspl)]) + 0.5 # subregion of the original image
            axins = ax.inset_axes(
                [0.64,0.64,0.35,0.33],
                xlim=(x1, x2), ylim=(y1, y2), xticks=[])
            axins2 = ax.inset_axes(
                [0.64,0.48,0.35,0.14],
                xlim=(x1, x2))
            axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)

            axins.plot(my_data.time, np.log10(psplfluxfin), color='black', linestyle=':', markersize=0.5)
            
            residualsplan = (my_data.flux - planetfluxfin)/my_data.flux
            axins.plot(my_data.time, np.log10(planetfluxfin), color='red', linestyle='--', markersize=0.5)
            axins2.scatter(my_data.time, residualsplan , color='red', s=0.5, label='Planet')
     
                
            handles, labels = axins2.get_legend_handles_labels()
            legend1 = axins2.legend(handles, labels, loc='best')
            axins2.add_artist(legend1)
            axins2.set_xlabel('HJD')
            axins2.set_ylabel('Residuals')
            plt.subplots_adjust(hspace=0.1)
            #Save figure
            plt.savefig(outpath+"u%.3falpha%d_plot.png"%(fu0,alpha))
            plt.close()
            end_time = time.time()
            print("Time taken for u0 = %.3f and alpha = %.1f is %.2f seconds"%(fu0,alpha,end_time-start_time))
    return


if __name__ == "__main__":
    print("Starting the program")
    start_time = time.time()
    falphal = [72.0]
    fu0l = [-0.20]
    #falpha, u0= np.loadtxt('../Multiple_lc_run1_fits/Binaryfit_fail_chisq.txt', usecols =(0,1), unpack = True, dtype=float)
  
    #create tuple pairs of alpha and u0
    """
    falphau0 = list(zip(falpha, u0))
    #remove duplicates
    falphau0 = list(set(falphau0))
    falphau01 = falphau0[:len(falphau0)//2]
    falphau02 = falphau0[len(falphau0)//2:]
    #fu0l = [-0.015]#[ -0.2, -0.1, -0.075, -0.050, -0.025, -0.015, 0.0, 0.015, 0.025, 0.05, 0.075, 0.1, 0.2]#[float(x) for x in input_list]

    #falphal = [120.0]#np.linspace(float(input_list[0]), float(input_list[1]), int(input_list[2]), endpoint=False)

    filenamein = "../"#input("Enter the filepath to Multiple_lc_run1: ")
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=task_manager, args=(filenamein, falphau01, q))
    p2 = multiprocessing.Process(target=task_manager, args=(filenamein, falphau02, q))
    p1.start()
    p2.start()

    result_1 = q.get()
    result_2 = q.get()

    p1.join()
    p2.join()

    fitfail_list = result_1 + result_2
    np.savetxt(filenamein+"Multiple_lc_run1_fits/Binary_fit_new2p1.txt", fitfail_list, fmt = "%s")
    """
    """
    for falpha in falphal:
        processes = []
        for fu0 in fu0l:
            p = multiprocessing.Process(target=task_manager, args=(filenamein, falpha, fu0))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    """
    filenamein = "../data_files"

    lockbinaryfail = multiprocessing.Lock()
    locknotcb = multiprocessing.Lock()
    lockmem = multiprocessing.Lock()
   
    task_manager(filenamein, falphal, fu0l, lockbinaryfail, locknotcb, lockmem)

    end_time = time.time()
    print("Total time taken is %.2f seconds"%(end_time-start_time))
    