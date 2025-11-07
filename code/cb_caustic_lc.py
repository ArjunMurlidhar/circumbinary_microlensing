import sys
sys.path.append('/Users/murlidhar.4/Documents/Projects/microlens/triplelens-1.0.7/test/')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
#import TripleLensing
#TRIL = TripleLensing.TripleLensing()
import MulensModel as mm
import matplotlib.lines as mlines
import numpy as np
import os
from time import time
import VBMicrolensing
import argparse


VBM = VBMicrolensing.VBMicrolensing()
VBM.Tol = 1e-4
VBM.SetMethod(VBM.Method.Singlepoly)

# Which direction??
# Planet orientation
def magtoflux(mag, zeropint = 27.615):
    return 10.**(-0.4*(mag - zeropint))

# sns.set_theme()
# sns.set_style("whitegrid", {'axes.grid' : True})

def gencblc(params, path):
    # meausre the time taken to run this function
    print("Started generating light curve")
    start = time()
    t0, u0, tE, s2, q2, alpha, s3, q3, psi, rs = params
    salpha = np.sin(alpha)
    calpha = np.cos(alpha)
    params = [t0, u0, tE, s2, q2, alpha, s3, q3, psi, rs]
    psideg = psi*180./np.pi
    # set up lens system
    m1 = 1./(1. + q2) #Normalized so that total mass of binary=1. VBMicrolensing - coordinates are in units of Einstein radius for a unitary mass lens
    m2 = q2 * m1
    m3 = q3 * m1
    z1x = -q2*s2/(1. + q2)
    z2x = s2/(1. + q2)
    z3x = -q2*s2/(1. + q2) + s3*np.cos(psi)
    z3y = s3*np.sin(psi)
    mlens = [m1, m2, m3]
    zlens = [z1x,0.,z2x,0.,z3x,z3y]
    number = "1"
    gamma = (75./2.)/tE
    # source position
    ts = np.linspace(t0 - gamma*tE, t0 + gamma*tE, int(2.*gamma*tE*1440./15.))
    tsdash = np.linspace(t0 - 5.*tE, t0 + 5.*tE, 10000)
    tn = (ts - t0) / tE  # tau
    tndash = (tsdash - t0) / tE
    y1s = u0 * salpha + tn * calpha  # source positions
    y2s = u0 * calpha - tn * salpha
    y1straj = u0 * salpha + tndash * calpha  # source positions for source trajectory
    y2straj = u0 * calpha - tndash * salpha

    #Initialize the lens system - VB microlensing
    parameters = [z1x,0,m1, # First lens: x1_1, x1_2, m1
              z2x,0,m2,     # Second lens: x2_1, x2_2, m2
              z3x,z3y,m3    # Third lens: x3_1, x3_2, m3
                        ]   # Fourth lens: x4_1, x4_2, m4

    VBM.SetLensGeometry(parameters) #Initialize the lens configuration

    #parameters controls the accuracy of finite source calculation
    secnum = 45  # divide the source bondary into how many parts
    basenum = 2  # the number density of sampled dots among each part
    quaderr_Tol = 1e-3  # the Quadrupole test tolerance
    relerr_Tol = 1e-3  # the relative error tolerance for magnification

    # planet orientation (planet at planet position, star at origin)
    splanet = np.sqrt(z3x**2 + z3y**2)
    qplanet = m3/(m1+m2)
    alphamulens = (2.0*np.pi - alpha)*180./np.pi
    planetcom = np.array([(m3*z3x)/(m1+m2+m3),(m3*z3y)/(m1+m2+m3)])  # Position of COM of planet and star
    planetcomdist = np.sqrt(planetcom[0]**2 + planetcom[1]**2)
    u0planet = np.abs(planetcom[0]*salpha + planetcom[1]*calpha - u0)*np.sign(u0)  # impact parameter of planet
    slopep = z3y/z3x
    slopet = -np.tan(alpha)
    beta = np.arctan(np.abs((slopep - slopet)/(1. + slopep*slopet)))
    #print(round(np.abs(u0planet - u0),6),round(planetcomdist*np.sin(beta),6))

    if z3y > 0:
        planetorient = np.arctan2(z3y,z3x)
    elif z3y == 0:
        if z3x > 0:
            planetorient = 0.
        elif z3x < 0:
            planetorient = np.pi
    else:
        planetorient = 2*np.pi + np.arctan2(z3y,z3x)
    alphaplanet = alphamulens - planetorient*180./np.pi


    """
    #algorithm to find alphaplanet
    m1 = z3y/z3x #slope of line passing through planet and origin
    m2 = -1./m1 #slope of line perpendicular to line passing through planet and origin
    if z3x < 0:
        deltat = 1
    else:
        deltat = -1
    n1 = np.array([deltat,m2*deltat]) #normal vector to line passing through planet and origin
    n2 = np.array([calpha*(calpha*planetcom[0] - salpha*planetcom[1]) + salpha*u0, salpha*(salpha*planetcom[1] - calpha*planetcom[0]) + calpha*u0]) #position of the base of the normal to trajectory from planetcom 
    v = n2 - planetcom
    beta = np.arccos(np.dot(v,n1)/(np.linalg.norm(v)*np.linalg.norm(n1))) #angle between normal to trajectory and normal to line passing through planet and origin
    if (-planetcom[1] + m2*planetcom[0])*(n2[1] - planetcom[1] - m2*(n2[0] - planetcom[0])) > 0:
        print("here")
        if u0 > 0:
            alphaplanet = beta
        else:
            alphaplanet = np.pi + beta
    else:
        if u0 > 0:
            alphaplanet = 2*np.pi - beta
        else:
            alphaplanet = np.pi - beta
    """
    print("splanet = %.4f, qplanet = %.5f, planetcomdist = %.5f, u0planet = %.5f, alphamulens = %.1f, planetorient = %.2f, alphaplanet = %.2f"%(splanet, qplanet, planetcomdist, u0planet, alphamulens, planetorient*180./np.pi, alphaplanet))

    #binary without planet and planetary system with star at COM of binary
    binary = mm.Model( {'t_0': t0, 'u_0': u0, 't_E': tE, 's': s2, 'q': q2, 'alpha': alphamulens,
        'rho': rs})
    #binary_caustic = mm.Caustics(q = q2, s = s2)
    #planet_caustic = mm.Caustics(q = qplanet, s = splanet)
    binary.set_magnification_methods([t0 - gamma*tE, 'VBBL', t0 + gamma*tE])
    binary.set_magnification_methods_parameters({'VBBL': {'accuracy': 1e-4}})
    planet = mm.Model( {'t_0': t0, 'u_0': u0planet, 't_E': tE, 's': splanet, 'q': qplanet, 'alpha': alphaplanet,
        'rho': rs})
    planet.set_magnification_methods([t0 - gamma*tE, 'VBBL', t0 + gamma*tE])
    planet.set_magnification_methods_parameters({'VBBL': {'accuracy': 1e-4}})
    pspl = mm.Model({'t_0': t0, 'u_0': u0, 't_E': tE})
    print("Created planet and binary models")

    """
    # Caustics for triple lens
    caustics = VBM.Multicaustics()
    f = plt.figure(figsize=(12,12),dpi=300)
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    plt.subplots_adjust(top = 0.9, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.1, wspace = 0.2)



    # plot lens position

    for i in range(3): #3
        ax.scatter(zlens[i*2], zlens[i*2+1], marker = "+",s = 200, color = "k")

    for cau in caustics:
        plt.plot(cau[0], cau[1], color = 'r', label = "CB")
    plt.axis("equal")
    plt.grid(True)
    #plot a line to show source trajectory
    ax.plot(y1straj,y2straj,color = "k", linestyle = "--", label = "Source trajectory")
    plt.xlim(-0.35,0.35)
    plt.ylim(-0.35,0.35)
    plt.legend(loc = "best")
    plt.title("s2 = %.3f, q2 = %.3f, s3 = %.3f, q3 = %.3f, psi = %.3f, rs = %.3f"%(s2,q2,s3,q3,psideg,rs))
    plt.savefig('../../Multiple_lc_run2-Me/%s/u%.3falpha%d_caustic.png'%(folder,u0,alphamulens))
    plt.close()
    
    #zoom into central caustic

    f = plt.figure(figsize=(12,12),dpi=300)
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    plt.subplots_adjust(top = 0.9, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.1, wspace = 0.2)

    # plot lens position
    for i in range(3): #3
        ax.scatter(zlens[i*2], zlens[i*2+1], marker = "+",s = 200, color = "k")

    for cau in caustics:
        plt.plot(cau[0], cau[1], color = 'r', label = "CB")
    plt.axis("equal")
    plt.grid(True)
    plt.xlim(-0.1,0.1)
    plt.ylim(-0.1,0.1)
    #plot a line to show source trajectory
    ax.plot(y1straj,y2straj,color = "k", linestyle = "--", label = "Source trajectory")
    plt.title("s2 = %.3f, q2 = %.3f, s3 = %.3f, q3 = %.3f, psi = %.3f, rs = %.3f"%(s2,q2,s3,q3,psideg,rs))
    binary_caustic.plot(color = "black", s = 1, label = "Binary")
    plt.legend(loc = "best")
    plt.savefig('../../Multiple_lc_run2-Me/%s/u%.3falpha%d_causticzoom.png'%(folder,u0,alphamulens))


    #zoom into binary secondary caustic

    f = plt.figure(figsize=(12,12),dpi=300)
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    plt.subplots_adjust(top = 0.9, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.1, wspace = 0.2)

    ax.scatter(causx,causy, marker='.', s= 1,color='r', label = 'CB')
    plt.axis("equal")
    plt.xlim(-0.22,-0.18)
    plt.ylim(-4.1,-4.0)
    #plot a line to show source trajectory
    ax.plot(y1straj,y2straj,color = "k", linestyle = "--")
    plt.title("s2 = %.3f, q2 = %.3f, s3 = %.3f, q3 = %.3f, psi = %.3f, rs = %.3f"%(s2,q2,s3,q3,psideg,rs))
    #binary_caustic.plot(color = "black", s = 1, label = "Binary caustic")
    #planet.plot_caustics(color = "green", s = 1, label = "Planet caustic")
    plt.legend(loc = "best")
    plt.savefig('./Plots/circumbinary/%s/caustic_cb_zoomsec%s.png'%(folder,number))



    #zoom into planetary secondary caustic

    f = plt.figure(figsize=(12,12),dpi=300)
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    plt.subplots_adjust(top = 0.9, bottom = 0.08, right = 0.95, left = 0.1, hspace = 0.1, wspace = 0.2)

    ax.scatter(causx,causy, marker='.', s= 1,color='r', label = 'CB')
    plt.axis("equal")
    plt.xlim(6.70,6.71) #-1,0
    plt.ylim(3.93, 3.94)#-1,0
    #plot a line to show source trajectory
    ax.plot(y1straj,y2straj,color = "k", linestyle = "--")
    plt.title("s2 = %.3f, q2 = %.3f, s3 = %.3f, q3 = %.3f, psi = %.3f, rs = %.3f"%(s2,q2,s3,q3,psideg,rs))
    #binary_caustic.plot(color = "black", s = 1, label = "Binary caustic")
    #ax.scatter(planetxr,planetyr, marker='.', s= 1,color='g', label = "Planet")
    #x = np.linspace(-0.2,0.05,100)
    #y = np.tan(psidash)*(x - 0.0229)
    #ax.plot(x,y,color = "blue", linestyle = "--")
    plt.legend(loc = "best")
    plt.savefig('./Plots/circumbinary/%s/caustic_cb_zoomplan%s.png'%(folder,number))
    """
    
    vbmag = []
    print("Started vbmag calculation")
    for i in range(len(y1s)):
        vbmag.append(VBM.MultiMag(y1s[i],y2s[i],rs))
    vbmag = np.array(vbmag)
    print("Generated CB LC")
    #mus = TRIL.TriLightCurve(mlens, zlens, y1s, y2s, rs, secnum, basenum, quaderr_Tol, relerr_Tol)
    photapert = 9
    exp_time = 50
    sourcemag = 20.0
    blendmag = 27.615
    sourceflux = magtoflux(sourcemag)
    blendflux = magtoflux(blendmag)
    rdnoise = 12.12*photapert
    darkcurr = 1.072*exp_time*photapert
    sky = 3.43*exp_time*photapert
    cbflux = sourceflux*np.array(vbmag) + blendflux
    #lcfluxerror = np.random.normal(cbflux, np.sqrt(cbflux + darkcurr + sky + rdnoise**2), len(cbflux))
    lcerrorbar = np.sqrt(cbflux + darkcurr + sky + rdnoise**2)
    planetmag = planet.get_magnification(ts)
    planetlc = sourceflux*np.array(planetmag) + blendflux
    #planetlcerr = np.random.normal(planetlc, np.sqrt(planetlc + darkcurr + sky + rdnoise**2), len(planetlc))
    planeterrorbar = np.sqrt(planetlc + darkcurr + sky + rdnoise**2)
    binarymag = binary.get_magnification(ts)
    binarylc = sourceflux*np.array(binarymag) + blendflux
    #binarylcerr = np.random.normal(binarylc, np.sqrt(binarylc + darkcurr + sky + rdnoise**2), len(binarylc))
    binlcerrorbar = np.sqrt(binarylc + darkcurr + sky + rdnoise**2)
    #psplmag = pspl.get_magnification(ts)
    #pspllc = sourceflux*np.array(psplmag) + blendflux
    """
    diffmagcbb = (cbflux - binarylc)/binarylc #lcfluxerror REPLACED
    diffmagcbp = (cbflux - planetlc)/planetlc
    diffmagcbpspl = (cbflux - pspllc)/pspllc
    """
    print("Generated planet and binary LC")
    np.savetxt(path+'/u%.3falpha%d_flux.txt'%(u0,alphamulens), np.array([ts,cbflux,lcerrorbar]).T)
    np.savetxt(path+'/u%.3falpha%d_info.txt'%(u0,alphamulens), np.array([t0,u0,tE,s2,q2,alphamulens,rs,s3,q3,psideg]), header='t0, u0, tE, s2, q2, alpha, rs, s3, q3, psi')

    np.savetxt(path+'/u%.3falpha%d_planetflux.txt'%(u0,alphamulens), np.array([ts,planetlc,planeterrorbar]).T)
    np.savetxt(path+'/u%.3falpha%d_planetinfo.txt'%(u0,alphamulens), np.array([t0,u0planet,tE,splanet,qplanet,alphaplanet,rs]), header='t0, u0, tE, s, q, alpha, rs')

    np.savetxt(path+'/u%.3falpha%d_binflux.txt'%(u0,alphamulens), np.array([ts,binarylc,binlcerrorbar]).T)
    np.savetxt(path+'/u%.3falpha%d_bininfo.txt'%(u0,alphamulens), np.array([t0,u0,tE,s2,q2,alphamulens,rs]), header='t0, u0, tE, s2, q2, alpha, rs')
    print("Saved files")
    
    """
    #Plot the CB light curve with binary, planet and pspl models
    fig, ax = plt.subplots(figsize = (15,10), dpi = 200)
    ax.errorbar(ts, np.log10(cbflux), yerr = lcerrorbar/(cbflux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5, label = r'Data : $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$s_2$=%0.2f,$q_2$=%0.3f,$\alpha$=%0.1f,$\rho$=%0.3f, $s_3$ = %0.2f, $q_3$ = %0.3f, $\psi$=%0.1f'%(t0, u0, tE, s2, q2, alphamulens, rs, s3, q3, psideg))
    ax.plot(ts, np.log10(pspllc), linestyle = ':', color = 'black', markersize=0.8, label= r'PSPL Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f'%(t0, u0, tE, rs))
    ax.plot(ts,np.log10(binarylc), linestyle = '-', color = 'blue', markersize=0.8, label= r'Binary Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(t0, u0, tE, rs, s2, q2, alphamulens))
    ax.plot(ts, np.log10(planetlc), color='red', linestyle='--', markersize=0.8, label= r'Planet Model: $t_0$=%.1f,$u_0$=%.3f,$t_E$=%.1f,$\rho$=%0.3f,s=%0.2f,q=%0.3f,$\alpha$=%0.1f'%(t0, u0planet,tE, rs, splanet, qplanet, alphaplanet))
    
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Log(Flux)")
    ax.set_title("Data with binary, planetary, and PSPL light curves")
    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc='lower center')
    ax.add_artist(legend1)

    residualpspl = np.abs(cbflux - pspllc)/cbflux
    tsmaxdev = ts[np.argmax(residualpspl)]

    #inset axes
    if np.max(residualpspl) < 0.1:
        x1, x2, y1, y2 = 2000. - 3, 2000. + 3, np.log10(np.max(cbflux)) - 0.5, np.log10(np.max(cbflux)) + 0.5  # subregion of the original image
    else:    
        x1, x2, y1, y2 = tsmaxdev - 3, tsmaxdev + 3, np.log10(cbflux[np.argmax(residualpspl)]) - 0.5, np.log10(cbflux[np.argmax(residualpspl)]) + 0.5  # subregion of the original image
    axins = ax.inset_axes(
        [0.64,0.64,0.35,0.33],
        xlim=(x1, x2), ylim=(y1, y2), xticks=[])
    axins2 = ax.inset_axes(
        [0.64,0.48,0.35,0.14],
        xlim=(x1, x2))
    axins.errorbar(ts, np.log10(cbflux), yerr=lcerrorbar/(cbflux*np.log(10)), fmt= '.', color='darkorange', markersize=0.5)

    axins.plot(ts, np.log10(pspllc), color='black', linestyle=':', markersize=0.5)
    residualsplan = (cbflux - planetlc)/cbflux
    axins.plot(ts, np.log10(planetlc), color='red', linestyle='--', markersize=0.5)
    axins2.scatter(ts, residualsplan , color='red', s=0.5, label='Planet')
    residualsbin = (cbflux - binarylc)/cbflux
    axins.plot(ts, np.log10(binarylc), color='blue', linestyle='-', markersize=0.5)
    axins2.scatter(ts, residualsbin, color='blue', s=0.5, label='Binary')
    
    handles, labels = axins2.get_legend_handles_labels()
    legend1 = axins2.legend(handles, labels, loc='best')
    axins2.add_artist(legend1)
    axins2.set_xlabel('HJD')
    axins2.set_ylabel('Residuals')
    plt.subplots_adjust(hspace=0.1)
    #main, gs = pltlkv(ts, mus, params, label = "Without limb-darkening")
    plt.savefig(path+'/u%.3falpha%d_lc.png'%(u0,alphamulens))
    plt.close()
    """
    end = time()
    print("Time taken = %.2f"%(end - start))


parser = argparse.ArgumentParser()
parser.add_argument('--input_s', type=float, required=True, help="Input value of s3")
parser.add_argument('--input_q', type=float, required=True, help="Input value of q3")
parser.add_argument('--input_psi', type=float, required=True, help="Input value of psi (in radians)")
parser.add_argument('--input_u', type=float, nargs='+', required=True, help="Input values of u0 (provide space-separated values)")
args = parser.parse_args()
#event parameters
t0 = 2000.
u0 = args.input_u 
tE = 30.
s2 = 0.20
q2 = 0.99
alpha = np.linspace(0.0, np.pi, 10, endpoint=False) #//rad
s3plan = args.input_s
q3plan = args.input_q
psi = args.input_psi #//rad
s3 = 0.1*np.cos(psi) + np.sqrt(0.01*(np.cos(psi)**2 - 1) + s3plan**2)
q3 = 2.*q3plan
rs = 0.005

for alphar in alpha:
    folder = "alpha%.1f"%(alphar*180./np.pi)
    path = '/home/murlidhar.4/Data/Planet_grid_run/data_files/s%.2fq%.1epsi%.1f/'%(s3plan,q3plan,psi*180./np.pi)+folder
    if not os.path.exists(path):
        os.makedirs(path)
    for u in u0:
        params = [t0, u, tE, s2, q2, alphar, s3, q3, psi, rs]
        gencblc(params, path)



#binary light curve
#plt.figure()
#binary.plot_magnification(t_range=(t0-5.*tE, t0+5.*tE), 
 #   color='red', linestyle='-', zorder=1)
#plt.title('u0 = %.3f, s = %0.2f, q = %0.3f, rho = %0.3f, alpha = %0.1f'%(u0,s2,q2,rs, alphamulens))
#plt.yscale('log')
#plt.savefig('./Plots/circumbinary/%s/lc_binary%s.png'%(folder,number))

#planet light curve
#plt.figure()
#planet.plot_magnification(t_range=(t0-5.*tE, t0+5.*tE), 
#    color='red', linestyle='-', zorder=1)
#plt.title('u0 = %.3f, s = %0.2f, q = %0.3f, rho = %0.3f, alpha = %0.1f'%(u0,splanet,qplanet,rs, alphamulens))
#plt.savefig('./Plots/circumbinary/%s/lc_planet%s.png'%(folder,number))



#diff between cb and binary

#plt.figure()
#plt.plot(time, diffmagcbb)
#plt.xlabel("Time")
#plt.ylabel("Fractional deviation")
#plt.title("Fractional deviation from binary model")
#plt.savefig('./Plots/circumbinary/%s/diff_cb_binary%s.png'%(folder,number))

#diff between cb and planet

#plt.figure()
#plt.plot(time, diffmagcbp)
#plt.xlabel("Time")
#plt.ylabel("Fractional deviation")
#plt.title("Fractional deviation from planet model")
#plt.savefig('./Plots/circumbinary/%s/diff_cb_planet%s.png'%(folder,number))

