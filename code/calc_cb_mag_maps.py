#calculate magnification maps
import magnification_maps as mag
import numpy as np
import argparse
#Function to calculate planet offset required to match the planetary part of the circumbinary caustic
def calculate_planet_offset(s3, q3, psi_deg, s2, q2):
    psi = np.deg2rad(psi_deg)

    # Masses normalized so that total binary mass = 1
    m1 = 1.0 / (1.0 + q2)
    m2 = q2 * m1
    m3 = q3 * m1

    # Lens positions (binary along x, planet offset by s3 at angle psi)
    z1x = -q2 * s2 / (1.0 + q2)
    z2x = s2 / (1.0 + q2)
    z3x = s3 * np.cos(psi) #-q2_fixed * s2_fixed / (1.0 + q2_fixed) + 
    z3y = s3 * np.sin(psi)

    qplanet = m3 / (m1 + m2)

    #Position of planet wrt mass 1 and mass 2
    pA = (z3x - z1x, z3y)
    pB = (z3x - z2x, z3y)
    alphaA = np.arctan2(pA[1], pA[0])
    alphaB = np.arctan2(pB[1], pB[0])
    sA = np.sqrt(pA[0]**2 + pA[1]**2)
    sB = np.sqrt(pB[0]**2 + pB[1]**2)
    qA = m3/m1
    qB = m3/m2
    #position of planet caustic due to mass 1 and mass 2
    rA = ((1-qA)/(1+qA))*(sA - (1./sA))
    rB = ((1-qB)/(1+qB))*(sB - (1./sB))
    rc1 = (z1x + rA*np.cos(alphaA), rA*np.sin(alphaA))
    rc2 = (z2x + rB*np.cos(alphaB), rB*np.sin(alphaB))
    #print(rc1, rc2)

    #Position of circumbinary planet caustic
    rcb = (1./(1+q2))*np.array(rc1) + (q2/(1+q2))*np.array(rc2)
    #print(rcb)
    rcb_norm = np.sqrt(rcb[0]**2 + rcb[1]**2)
    c = rcb_norm*(1 + qplanet)/(1 - qplanet)
    if s3 < 1:
        splanet = np.abs((c - np.sqrt(c**2 + 4))/2)
        psi_new = np.arctan2(-rcb[1], -rcb[0])
    else:
        splanet = (c + np.sqrt(c**2 + 4))/2
        psi_new = np.arctan2(rcb[1], rcb[0])
    
    #print(psi_new)
    if psi_new < 0:
        psi_new += 2*np.pi
    psi_new_deg = np.rad2deg(psi_new)
    return splanet, qplanet, psi_new_deg

if __name__ == "__main__":
    #Accept user input for parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=float, nargs='*', required=True, help="Input value of circumbinary parameters (s2, q2, s3, q3, psi (deg), rho) in that order")
    parser.add_argument('--side', type=float, required=True, help="Input value of side length of the grid")
    parser.add_argument('--res', type=float, required=False, default=1e-3, help="Input value of resolution of the grid")
    parser.add_argument('--num_cores', type=int, required=False, default=1, help="Input value of number of cores to use for parallel computation")
    parser.add_argument('--float_precision', type=str, required=False, default='float32', help="Input value of float precision to be used (float32 or float64)")
    parser.add_argument('--backend', type=str, required=False, default='auto', help="Input value of backend for parallel computation (auto, processes, threads)")
    parser.add_argument('--outfile_suffix', type=str, required=True, help="Input value of output file suffix")
    parser.add_argument('--outfolder', type=str, required=True, help="Input value of output folder")
    parser.add_argument('--compression', type=str, required=False, default='GZIP_2', help="Input value of output compression format (GZIP_2, RICE_1, PLIO_1, HCOMPRESS_1)")
    parser.add_argument('--binary_diff', action='store_true', help="Flag to calculate the binary difference map")
    parser.add_argument('--planet_diff', action='store_true', help="Flag to calculate the planet difference map")
    
    args = parser.parse_args()
    params = args.params
    side = args.side
    res = args.res
    num_cores = args.num_cores
    float_precision = args.float_precision
    backend = args.backend
    outfile_suffix = args.outfile_suffix
    outfolder = args.outfolder
    compression = args.compression
    binary_diff = args.binary_diff
    planet_diff = args.planet_diff
    
    #parameters for the magnification map
    params_cb = {
        's2': params[0],
        'q2': params[1],
        's3': params[2],
        'q3': params[3],
        'psi': params[4],
        'rho': params[5]
    }
    param_s = {
        'rho': params[5]
    }

    cb_map = mag.create_mag_map(params_cb, res=res, side=side, num_cores=num_cores, float_precision=float_precision, backend=backend)
    sing_map = mag.create_mag_map(param_s, res=res, side=side, num_cores=num_cores, float_precision=float_precision, backend=backend)
    cb_sing_map = (cb_map - sing_map)/sing_map
    mag.save_mag_map(cb_sing_map, outfolder+'/cb_sing_map_'+outfile_suffix+'.fits.fz', compression=compression, metadata={'s2': params[0], 'q2': params[1], 's3': params[2], 'q3': params[3], 'psi': params[4], 'rho': params[5], 'res': res, 'side': side, 'dtype': str(cb_sing_map.dtype)})
    if binary_diff:
        param_b = {
            's': params[0],
            'q': params[1],
            'rho': params[5]
        }
        bin_map = mag.create_mag_map(param_b, res=res, side=side, num_cores=num_cores, float_precision=float_precision, backend=backend)
        cb_bin_map = (cb_map - bin_map)/bin_map
        mag.save_mag_map(cb_bin_map, outfolder+'/cb_bin_map_'+outfile_suffix+'.fits.fz', compression=compression, metadata={'s2': params[0], 'q2': params[1], 's3': params[2], 'q3': params[3], 'psi': params[4], 'rho': params[5], 'res': res, 'side': side, 'dtype': str(cb_bin_map.dtype)})
    if planet_diff:
        splanet, qplanet, psi_planet = calculate_planet_offset(params[2], params[3], params[4], params[0], params[1])
        param_p = {
            's': splanet,
            'q': qplanet,
            'psi': psi_planet,
            'rho': params[5]
        }
        planet_map = mag.create_mag_map(param_p, res=res, side=side, num_cores=num_cores, float_precision=float_precision, backend=backend)
        cb_planet_map = (cb_map - planet_map)/planet_map
        mag.save_mag_map(cb_planet_map, outfolder+'/cb_planet_map_'+outfile_suffix+'.fits.fz', compression=compression, metadata={'s2': params[0], 'q2': params[1], 's3': params[2], 'q3': params[3], 'psi': params[4], 'rho': params[5], 'splanet': splanet, 'qplanet': qplanet, 'psi_planet': psi_planet, 'res': res, 'side': side, 'dtype': str(cb_planet_map.dtype)})

    print("Calculated magnification maps and saved to "+outfolder)



