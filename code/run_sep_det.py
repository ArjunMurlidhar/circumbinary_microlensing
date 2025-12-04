import numpy as np
from calc_det_sep import region_of_dev, calc_detectability, refine_det
import argparse
import csv
import os
import ast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="Run name")
    parser.add_argument("--input_file", type=str, required=True, help="Input file name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_cores", type=int, default = 1, help="Number of cores to use")
    parser.add_argument("--mag_plot", action="store_true", help="Plot magnification map")
    args = parser.parse_args()

    run_name = args.run_name
    input_file = args.input_file
    output_dir = args.output_dir
    num_cores = args.num_cores
    mag_plot = args.mag_plot

    
    # Input file columns: s2, q2, s3, q3, psi, rho, tE, cad, bin_box, planet_box, contour_threshold, alpha_density, n_pot
    # Read input file and store each row as a dictionary with keys as the column names
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        fixed_bin_params = []
        fixed_bin_index = 0
        sys_ind = 0
        results = []
        for row in reader:
            print("Running system ", run_name, sys_ind)
            sys = run_name + "_" + str(sys_ind)
            row_f = {k: float(v) for k, v in row.items()}
            params = {
                "s2": row_f["s2"],
                "q2": row_f["q2"],
                "s3": row_f["s3"],
                "q3": row_f["q3"],
                "psi": row_f["psi"],
                "rho": row_f["rho"],
            }
            bin_params = [params["s2"], params["q2"], params["rho"], row_f["cad"], row_f["tE"]]
            if len(fixed_bin_params) == 0 or bin_params != fixed_bin_params:
                fixed_bin_params = bin_params
                fixed_bin_index = sys_ind
                #Run region of deviation
                #if bin_map_contours.txt and planet_map_contours.txt exist, get bin_u0 and planet_corners from them
                if os.path.exists(os.path.join(output_dir, sys, "bin_map_contours.txt")) and os.path.exists(os.path.join(output_dir, sys, "planet_map_contours.txt")):
                    with open(os.path.join(output_dir, sys, "bin_map_contours.txt"), 'r') as f:
                        for line in f:
                            if line.startswith("# Bin_u0:"):
                                bin_u0 = float(line.split(":")[1].strip())   
                    with open(os.path.join(output_dir, sys, "planet_map_contours.txt"), 'r') as f:
                        for line in f:
                            if line.startswith("# Planet rectangle corners:"):
                                planet_corners = line.split(":")[1].strip()
                    planet_corners = ast.literal_eval(planet_corners)
                    
                else:
                    bin_u0, planet_corners = region_of_dev(params, output_dir, sys, row_f["tE"], row_f["cad"], row_f["bin_box"], row_f["planet_box"], num_cores, row_f["contour_threshold"], mag_plot)
                if planet_corners is None:
                    results.append([sys_ind, 0, 0, 'NA', 'NA', 0, 0])
                    sys_ind += 1
                    continue
                res = row_f["cad"]/(row_f["tE"]*24*60)

                #run detectability
                n_det, n_u, n_alpha = calc_detectability(bin_u0, planet_corners, res, sys, output_dir, row_f["alpha_density"])

                #save a txt file with all trajectories inside bin_u0
                bin_u = np.arange(-bin_u0, bin_u0, res)
                if n_alpha > 360:
                    bin_alpha = np.linspace(0.0, np.pi, 360, endpoint=False)
                else:
                    bin_alpha = np.linspace(0.0, np.pi, n_alpha, endpoint=False)
                det_traj_bin = []
                for u in bin_u:
                    for alpha in bin_alpha:
                        det_traj_bin.append((u, alpha))

                bin_u = np.array([traj[0] for traj in det_traj_bin])
                bin_alpha = np.array([traj[1] for traj in det_traj_bin])
                np.savetxt(os.path.join(output_dir, sys, "det_traj_bin.txt"), np.array([bin_u, np.degrees(bin_alpha)]).T, fmt = '%.4f %.2f', header='u0  alpha(deg)')
                        
                #refine detectability
                n_det_ref = refine_det(output_dir, sys, threshold=row_f["contour_threshold"], n_pot=row_f["n_pot"], num_cores=num_cores)
                n_det_bin = refine_det(output_dir, sys, threshold=row_f["contour_threshold"], n_pot=row_f["n_pot"], type = 'bin', num_cores=num_cores)
                if n_alpha > 360:
                    n_det_bin = n_det_bin*n_alpha/360

                #calculate total number of trajectories
                total_traj = n_u*n_alpha

                bin_det_rate = n_det_ref/n_det_bin

                total_det_rate = n_det_ref/(total_traj*3.0/bin_u0)
                results.append([sys_ind, n_det, n_det_ref, n_det_bin, total_traj, bin_det_rate, total_det_rate])


            elif bin_params == fixed_bin_params:
                #if planet_map_contours.txt exist, get planet_corners from it
                if os.path.exists(os.path.join(output_dir, sys, "planet_map_contours.txt")):
                    with open(os.path.join(output_dir, sys, "planet_map_contours.txt"), 'r') as f:
                        for line in f:
                            if line.startswith("# Planet rectangle corners:"):
                                planet_corners = line.split(":")[1].strip()
                    #convert planet_corners to a list of tuples
                    planet_corners = ast.literal_eval(planet_corners)
                else:
                    bin_u0, planet_corners = region_of_dev(params, output_dir, sys, row_f["tE"], row_f["cad"], row_f["bin_box"], row_f["planet_box"], num_cores, row_f["contour_threshold"], mag_plot, skip_binary=True)
                if planet_corners is None:
                    results.append([sys_ind, 0, 0, 'NA', 'NA', 0, 0])
                    sys_ind += 1
                    continue
                if not os.path.exists(os.path.join(output_dir, sys, "bin_map_contours.txt")):
                    #create links to outdir/run_name_fixed_bin_index/bin_map_contours.txt, outdir/run_name_fixed_bin_index/bin_map.fits and outdir/run_name_fixed_bin_index/det_traj_bin.txt
                    os.symlink(os.path.join(output_dir, run_name + "_" + str(fixed_bin_index), "bin_map_contours.txt"), os.path.join(output_dir, sys, "bin_map_contours.txt"))
                    os.symlink(os.path.join(output_dir, run_name + "_" + str(fixed_bin_index), "bin_map.fits"), os.path.join(output_dir, sys, "bin_map.fits"))

                #Get bin_u0 from bin_map_contours.txt
                with open(os.path.join(output_dir, sys, "bin_map_contours.txt"), 'r') as f:
                    for line in f:
                        if line.startswith("# Bin_u0:"):
                            bin_u0 = float(line.split(":")[1].strip())
                            break
                res = row_f["cad"]/(row_f["tE"]*24*60)

                #run detectability
                n_det, n_u, n_alpha = calc_detectability(bin_u0, planet_corners, res, sys, output_dir, row_f["alpha_density"])

                #save a txt file with all trajectories inside bin_u0
                bin_u = np.arange(-bin_u0, bin_u0, res)
                det_traj_bin = []
                if n_alpha > 360:
                    bin_alpha = np.linspace(0.0, np.pi, 360, endpoint=False)
                else:
                    bin_alpha = np.linspace(0.0, np.pi, n_alpha, endpoint=False)
                for u in bin_u:
                    for alpha in bin_alpha:
                        det_traj_bin.append((u, alpha))

                bin_u = np.array([traj[0] for traj in det_traj_bin])
                bin_alpha = np.array([traj[1] for traj in det_traj_bin])
                np.savetxt(os.path.join(output_dir, sys, "det_traj_bin.txt"), np.array([bin_u, np.degrees(bin_alpha)]).T, fmt = '%.4f %.2f', header='u0  alpha(deg)')

                #refine detectability
                n_det_ref = refine_det(output_dir, sys, threshold=row_f["contour_threshold"], n_pot=row_f["n_pot"], num_cores=num_cores)
                n_det_bin = refine_det(output_dir, sys, threshold=row_f["contour_threshold"], n_pot=row_f["n_pot"], type = 'bin', num_cores=num_cores)
                if n_alpha > 360:
                    n_det_bin = n_det_bin*n_alpha/360

                #calculate total number of trajectories
                total_traj = n_u*n_alpha

                bin_det_rate = n_det_ref/n_det_bin

                total_det_rate = n_det_ref/(total_traj*3.0/bin_u0)

                results.append([sys_ind, n_det, n_det_ref, n_det_bin, total_traj, bin_det_rate, total_det_rate])

            sys_ind += 1
    
    #save results to a csv file
    with open(os.path.join(output_dir, run_name + "_results.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["sys_ind", "n_det", "n_det_ref", "n_det_bin", "total_traj", "bin_det_rate", "total_det_rate"])
        writer.writerows(results)
    





    


