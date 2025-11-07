import matplotlib.lines as mlines
import numpy as np
import matplotlib.pyplot as plt
import MulensModel as mm
import scipy.optimize as op
import os
import astropy.units as u

def plot_init_final(init_params, final_params, my_data, data_params, chisq, binplan, alpha):
    fig, ax = plt.subplots(figsize = (15,10), dpi = 200) 

    #Plot Data
    if len(data_params) == 7:
        dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s$=%0.2f,$q$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f'%(data_params["t_0"],data_params["u_0"],data_params["t_E"],data_params["s"],data_params["q"],data_params["alpha"],data_params["rho"]))
        ax.set_xlabel('HJD')
        ax.set_ylabel('log(Flux)')
        ax.set_title('u0 = %.3f, alpha = %.1f, %s'%(data_params["u_0"], alpha, binplan))
        plt.legend(loc='best')
    elif len(data_params) > 7:
        dat = ax.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(data_params["t_0"],data_params["u_0"],data_params["t_E"],data_params["s2"],data_params["q2"],data_params["alpha"],data_params["rho"], data_params["s3"], data_params["q3"], data_params["psi"]))
        ax.set_xlabel('HJD')
        ax.set_ylabel('log(Flux)')
        ax.set_title('u0 = %.3f, alpha = %.1f'%(data_params["u_0"], alpha))
        plt.legend(loc='best')
   
    #Define Models 
    init_model = mm.Model({'t_0': init_params["t_0"], 'u_0': init_params["u_0"], 't_E': init_params["t_E"], 's': init_params["s"], 'q': init_params["q"], 'alpha': init_params["alpha"], 'rho': init_params["rho"]})
    fin_model = mm.Model({'t_0': final_params["t_0"], 'u_0': final_params["u_0"], 't_E': final_params["t_E"], 's': final_params["s"], 'q': final_params["q"], 'alpha': final_params["alpha"], 'rho': final_params["rho"]})
    init_model.set_magnification_methods([init_params["t_0"] - 1.25*init_params["t_E"], 'VBBL', init_params["t_0"] + 1.25*init_params["t_E"]])   
    fin_model.set_magnification_methods([final_params["t_0"] - 1.25*final_params["t_E"], 'VBBL', final_params["t_0"] + 1.25*final_params["t_E"]])
    
    #Find fluxes and calculate light curves
    init_fit = mm.FitData(dataset=my_data, model=init_model)
    init_fit.fit_fluxes()
    initsourceflux = init_fit.source_flux #init_params["sf"]#
    initblendflux = init_fit.blend_flux #init_params["bf"]#
    initlcmag = init_model.get_lc(my_data.time, source_flux=initsourceflux, blend_flux=initblendflux)
    initlcflux = mm.Utils.get_flux_from_mag(initlcmag)

    fin_fit = mm.FitData(dataset=my_data, model=fin_model)
    fin_fit.fit_fluxes()
    finsourceflux = fin_fit.source_flux #final_params["sf"]#
    finblendflux = fin_fit.blend_flux #final_params["bf"]#
    finlcmag = fin_model.get_lc(my_data.time, source_flux=finsourceflux, blend_flux=finblendflux)
    finlcflux = mm.Utils.get_flux_from_mag(finlcmag)

    #Plot Models                                         
    ax.plot(my_data.time, np.log10(initlcflux), color='red', linestyle='--', markersize=0.5, label = r'Initial Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,$s$=%0.2f,$q$=%0.4f,$\alpha$=%0.1f'%(init_params["t_0"],init_params["u_0"],init_params["t_E"],init_params["rho"],init_params["s"],init_params["q"],init_params["alpha"]))
    mod = ax.plot(my_data.time, np.log10(finlcflux), color='blue', linestyle='-', markersize=0.5, label= r'Final Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,$s$=%0.2f,$q$=%0.4f,$\alpha$=%0.1f'%(final_params["t_0"],final_params["u_0"],final_params["t_E"],final_params["rho"],final_params["s"],final_params["q"],final_params["alpha"]))
    blue_line = mlines.Line2D([], [], color='blue', label='Delta Chi2 = %.1f'%(chisq))
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc='lower center')
    ax.add_artist(legend1)
    legend2 = ax.legend(handles = [blue_line], loc='upper left')
    ax.add_artist(legend2)

    #find residuals
    residuals = (my_data.flux - finlcflux)/my_data.flux
    tsmaxdev = my_data.time[np.argmax(residuals)]

    #inset axes
    if np.max(residuals) < 0.1:
        x1, x2, y1, y2 = 2000. - 3, 2000. + 3, np.log10(np.max(my_data.flux)) - 0.5, np.log10(np.max(my_data.flux)) + 0.5  # subregion of the original image
    else:    
        x1, x2, y1, y2 = tsmaxdev - 3, tsmaxdev + 3, np.log10(my_data.flux[np.argmax(residuals)]) - 0.5, np.log10(my_data.flux[np.argmax(residuals)]) + 0.5  # subregion of the original image
  # subregion of the original image
    axins = ax.inset_axes(
        [0.64,0.64,0.35,0.33],
        xlim=(x1, x2), ylim=(y1, y2), xticks=[])
    axins.errorbar(my_data.time, np.log10(my_data.flux), yerr=my_data.err_flux/(my_data.flux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)
    axins.plot(my_data.time, np.log10(initlcflux), color='red', linestyle='--', markersize=0.5)
    axins.plot(my_data.time, np.log10(finlcflux), color='blue', linestyle='-', markersize=0.5)
    #ax.indicate_inset_zoom(axins, edgecolor="black")

    
    axins2 = ax.inset_axes(
        [0.64,0.48,0.35,0.14],
        xlim=(x1, x2))
    axins2.scatter(my_data.time, residuals , color='red', label = 'Residuals', s=0.5)
    axins2.set_xlabel('HJD')
    axins2.set_ylabel('Residuals')
    plt.subplots_adjust(hspace=0.1)
    plt.savefig('../Multiple_lc_run2-Me/u%.3falpha%d_%s_test.jpg'%(data_params["u_0"], alpha, binplan))
    plt.close()
    return

if __name__ == "__main__":
    print("Running Plop_binary_models.py")
    falphal, fu0l, binplan = np.loadtxt("../Multiple_lc_run1_fits/Binaryfit_fail.txt", usecols=[0,1,2], unpack=True, dtype=str)
    deltachisq = []
    for i in range(len(falphal)):

        falpha = float(falphal[i])
        alpha = 360.0 - falpha
        fu0 = float(fu0l[i])
        file_name = "../Multiple_lc_run1/s20p2s31p0psi60.0alpha%.1f/u%.3falpha%d"%(falpha,fu0,alpha)
        table = []
        with open("../Multiple_lc_run1_fits/s20p2s31p0psi60.0alpha%.1f/u%.3falpha%d.txt"%(falpha,fu0,alpha)) as f:
            # reading each line
            j = 0    
            for line in f:
                #if i==0:
                    #continue
                table.append([])
                # reading each word        
                for word in line.split():
                    table[j].append(word)
                j +=1

        if binplan[i] == "Binary":
            #Read binary data
            hjd, binaryflux, binaryflux_err = np.loadtxt(file_name+"_binflux.txt", usecols=[0,1,2], unpack=True)
            my_data_bin = mm.MulensData([hjd, binaryflux, binaryflux_err], phot_fmt="flux", chi2_fmt="flux")
            true_params = np.loadtxt(file_name+"_bininfo.txt", skiprows=1)
            params_bin = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s": true_params[3], "q": true_params[4], "alpha": true_params[5], "rho": true_params[6]}
            for row in table:
                if ('Binary-Binary' in row and 'Initial' in row) or ('Initial-Binary' in row and 'only' in row):
                    bi = row
                if ('Binary-Binary' in row and 'Final' in row) or ('Final-Binary' in row and 'only' in row):
                    bf = row
            
        elif binplan[i] == "Planet":
            #Read planet data
            hjd, planetflux, planetflux_err = np.loadtxt(file_name+"_planetflux.txt", usecols=[0,1,2], unpack=True)
            my_data_bin = mm.MulensData([hjd, planetflux, planetflux_err], phot_fmt="flux", chi2_fmt="flux")
            true_params = np.loadtxt(file_name+"_planetinfo.txt", skiprows=1)
            params_bin = {"t_0": true_params[0], "u_0": true_params[1], "t_E": true_params[2], "s": true_params[3], "q": true_params[4], "alpha": true_params[5], "rho": true_params[6]}
            for row in table:
                if ('Planet-Planet' in row and 'Initial' in row) or ('Initial-Planet' in row and 'only' in row):
                    bi = row
                if ('Planet-Planet' in row and 'Final' in row) or ('Final-Planet' in row and 'only' in row):
                    bf = row


        params_bininit = {'t_0': float(bi[0]), 'u_0': float(bi[1]), 't_E': float(bi[2]), 's': float(bi[4]), 'q': float(bi[5]), 'alpha': float(bi[6]), 'rho': float(bi[3]), 'sf': float(bi[7]), 'bf': float(bi[8])}
        params_binfinal = {'t_0': float(bf[0]), 'u_0': float(bf[1]), 't_E': float(bf[2]), 's': float(bf[4]), 'q': float(bf[5]), 'alpha': float(bf[6]), 'rho': float(bf[3]), 'sf': float(bf[7]), 'bf': float(bf[8])}
        chisq = float(bf[9])
        #deltachisq.append(chisq)
        plot_init_final(params_bininit, params_binfinal, my_data_bin, params_bin, chisq, binplan[i], alpha)

        print(binplan[i], alpha, fu0)
#np.savetxt("../Multiple_lc_run1_fits/Binaryfit_fail_chisq.txt", np.transpose([falphal, fu0l, binplan, deltachisq]), fmt='%s')