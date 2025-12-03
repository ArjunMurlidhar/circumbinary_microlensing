from magnification_maps import create_mag_map, save_mag_map
import numpy as np
import matplotlib.pyplot as plt
import os
from calc_cb_mag_maps import calculate_planet_offset
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, as_completed

def region_of_dev(params, outdir, sys_ind, tE = 10, cad = 12, bin_box=1.0, planet_box=1.0, num_cores=8, contour_threshold=0.05, make_plots=True, skip_binary=False):
    """
    Code to calculate the size of regions around the binary and planet caustics with magnification deviations greater than a threshold 
    (only for planets in wide or close topologies).
    params: Dictionary of parameters for the system (s2, q2, s3, q3, psi, rho)
    tE: Einstein ring crossing time in days
    cad: Cadence of the observations in minutes
    bin_box: Box size for magnification map calculation of binary caustics
    planet_box: Box size for magnification map calculation of planet caustics
    """
    s2, q2, s3, q3, psi, rho = params['s2'], params['q2'], params['s3'], params['q3'], params['psi'], params['rho']
    splanet, qplanet, psi_planet = calculate_planet_offset(s3, q3, psi, s2, q2) #effective planet parameters
    #check if parameters are valid
    if qplanet > 0.01:
        raise ValueError("Planet mass ratio must be less than 0.01")
    dw = 1. + (3*qplanet**(1./3.)/2)
    dc = 1. - (3*qplanet**(1./3.)/4)
    if splanet > dc and splanet < dw:
        raise ValueError("Planet must be in either wide or close topology")
    #Create output directory if it doesn't exist
    sys_dir = os.path.join(outdir, sys_ind)
    if not os.path.exists(sys_dir):
        os.makedirs(sys_dir)
    tE = tE*24*60 #minutes
    res = cad/tE
    if not skip_binary:
        # Calculate magnification map for the central binary
        params_b = {'s': s2, 'q': q2, 'rho': rho}
        bin_map = create_mag_map(params_b, res=res, side=bin_box, num_cores=num_cores, float_precision='float32', backend='auto')
        sing_map = create_mag_map({'rho': rho}, res=res, side=bin_box, num_cores=num_cores, float_precision='float32', backend='auto')
        bin_map = (bin_map - sing_map)/sing_map
        save_mag_map(bin_map, os.path.join(sys_dir, 'bin_map.fits'), metadata={'s2': s2, 'q2': q2, 'rho': rho, 's3': s3, 'q3': q3, 'psi': psi, 'splanet': splanet, 'qplanet': qplanet, 'psi_planet': psi_planet, 'bin_box': bin_box})
        bin_map_abs = np.abs(bin_map)

        # Calculate contours
        x = np.linspace(-bin_box/2, bin_box/2, bin_map.shape[1])
        y = np.linspace(-bin_box/2, bin_box/2, bin_map.shape[0])
        X, Y = np.meshgrid(x, y)

        #Create a two panel plot for binary map and planet map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        #plot
        if make_plots:
            im = ax1.imshow(bin_map, origin='lower', extent=[-bin_box/2, bin_box/2, -bin_box/2, bin_box/2], cmap='viridis')     
            # Overlay shaded region for mask > 0.05
            ax1.contourf(X, Y, bin_map_abs, levels=[contour_threshold, np.max(bin_map_abs)], colors='white', alpha=0.3)
            plt.colorbar(im, ax=ax1, label='Fractional difference',)

        cb = ax1.contour(X, Y, bin_map_abs, levels=[contour_threshold], colors='red', linewidths=1.0)
        #Find the point on the contour that is farthest from the origin
        max_dist = 0
        max_dist_point = None
        for i, segment in enumerate(cb.allsegs[0]):
            for point in segment:
                dist = np.linalg.norm(point)
                if dist > max_dist:
                    max_dist = dist
                    max_dist_point = point
        bin_u0 = np.sqrt(max_dist_point[0]**2 + max_dist_point[1]**2)

        if make_plots:
            #plot circle of radius bin_u0
            ax1.add_patch(plt.Circle((0, 0), bin_u0, color='green', fill=False, linewidth=1.0))
        
        #Save contour coordinates
        contour_path = os.path.join(sys_dir, 'bin_map_contours.txt')
        with open(contour_path, 'w') as f:
            f.write(f"# Contours for {sys_ind}\n")
            f.write(f"# Binary map contours\n")
            f.write(f"# Parameters: s2={s2}, q2={q2}, rho={rho}, s3={s3}, q3={q3}, psi={psi}\n")
            f.write(f"# Bin_u0: {bin_u0}\n")
            f.write("# Segment_ID, X, Y\n")
            for i, segment in enumerate(cb.allsegs[0]):
                for point in segment:
                    f.write(f"{i}, {point[0]:.6f}, {point[1]:.6f}\n")

    if skip_binary:
        bin_u0 = None
        fig, ax2 = plt.subplots(1, 1, figsize=(5, 5))

    # Calculate magnification map for the planet
    
    
    x_caustic = ((1-qplanet)/(1+qplanet))*(splanet - (1./splanet))
    l_central = 4*qplanet/(splanet - (1./splanet))**2
    if np.abs(x_caustic) - planet_box/2 > l_central:
        tau1 = x_caustic - planet_box/2
        tau2 = x_caustic + planet_box/2
    else:
        if x_caustic > 0:
            tau1 = l_central #size of central caustic
            tau2 = x_caustic + planet_box/2
        else:
            tau1 = x_caustic - planet_box/2
            tau2 = -l_central #size of central caustic
    box_dim = [(tau1, tau2), (-planet_box/2, planet_box/2)]
    params_p = {'s': splanet, 'q': qplanet, 'rho': rho}
    planet_map = create_mag_map(params_p, res=res, box_dim=box_dim, num_cores=num_cores, float_precision='float32', backend='auto')
    sing_map = create_mag_map({'rho': rho}, res=res, box_dim=box_dim, num_cores=num_cores, float_precision='float32', backend='auto')
    planet_map = (planet_map - sing_map)/sing_map
    save_mag_map(planet_map, os.path.join(sys_dir, 'planet_map.fits'), metadata={'s2': s2, 'q2': q2, 's3': s3, 'q3': q3, 'psi': psi, 'rho': rho, 'splanet': splanet, 'qplanet': qplanet, 'psi_planet': psi_planet, 'tau1': tau1, 'tau2': tau2, 'planet_box': planet_box})
    planet_map_abs = np.abs(planet_map)
    # Calculate contours
    x = np.linspace(tau1, tau2, planet_map.shape[1])
    y = np.linspace(-planet_box/2, planet_box/2, planet_map.shape[0])
    X, Y = np.meshgrid(x, y)
    #plot
    cp = ax2.contour(X, Y, planet_map_abs, levels=[contour_threshold], colors='white', linewidths=1.0)
    if len(cp.allsegs[0][0]) == 0:
        print("No contour found")
        plt.close()
        return bin_u0, None
    if make_plots:
        im = ax2.imshow(planet_map, origin='lower', extent=[tau1, tau2, -planet_box/2, planet_box/2], cmap='viridis')
        # Overlay shaded region for mask > 0.05
        ax2.contourf(X, Y, planet_map_abs, levels=[contour_threshold, np.max(planet_map_abs)], colors='white', alpha=0.3)
        plt.colorbar(im, ax=ax2, label='Fractional difference')

    #Find the point along the Y-axis that is farthest from the origin
    max_dist = 0
    max_dist_point = None
    for i, segment in enumerate(cp.allsegs[0]):
        for point in segment:
            dist = np.abs(point[1])
            if dist > max_dist:
                max_dist = dist
                max_dist_point = point
    planet_top = max_dist_point[1]
    #Find the right and left edges of the caustic magnification map
    min_dist_left = np.inf
    min_dist_right = np.inf
    min_dist_left_point = None
    min_dist_right_point = None
    for i, segment in enumerate(cp.allsegs[0]):
            for point in segment:
                if np.abs(point[1]) < res and np.abs(point[0] - x_caustic) < min_dist_left and point[0] < x_caustic:
                    min_dist_left = np.abs(point[0] - x_caustic)
                    min_dist_left_point = point[0]
                if np.abs(point[1]) < res and np.abs(point[0] - x_caustic) < min_dist_right and point[0] > x_caustic:
                    min_dist_right = np.abs(point[0] - x_caustic)
                    min_dist_right_point = point[0]

    #Check if central deviation is missing 
    # Find the index in X (x-axis) closest to x_caustic
    ix = np.argmin(np.abs(x - x_caustic))
    # Find the index in Y (y-axis) closest to 0
    iy = np.argmin(np.abs(y - 0))
    # Select the value from the (iy, ix) position
    if planet_map_abs[iy, ix] < contour_threshold: #No central deviation, use leftmost edge of contour
        #find the point on the contour who's x-coordinate is farthest from the origin and closest to the origin
        max_dist = 0
        max_dist_point = None
        min_dist = np.inf
        min_dist_point = None
        for i, segment in enumerate(cp.allsegs[0]):
            for point in segment:
                dist = np.abs(point[0])
                if dist > max_dist:
                    max_dist = dist
                    max_dist_point = point
                if dist < min_dist:
                    min_dist = dist
                    min_dist_point = point
        if x_caustic > 0:
            planet_left = min_dist_point[0]
            planet_right = max_dist_point[0]
        else:
            planet_left = max_dist_point[0]
            planet_right = min_dist_point[0]

    else:
        planet_left = min_dist_left_point
        planet_right = min_dist_right_point

    if planet_left is None:
        planet_left = tau1
    if planet_right is None:
        planet_right = tau2
    #Find corners of the rectangle
    if x_caustic > 0:
        planet_corners = [(planet_left, np.abs(planet_top)), (planet_right, np.abs(planet_top)), (planet_right, -np.abs(planet_top)), (planet_left, -np.abs(planet_top))]
    else:
        planet_corners = [(planet_right, np.abs(planet_top)), (planet_left, np.abs(planet_top)), (planet_left, -np.abs(planet_top)), (planet_right, -np.abs(planet_top))]
    #plot rectangle of corners
    if make_plots:
        ax2.add_patch(plt.Polygon(planet_corners, color='green', fill=False, linewidth=1.0))
        plot_path = os.path.join(sys_dir, f"{sys_ind}_contour_plot.png")
        plt.savefig(plot_path)
        plt.close()

    #Save contour coordinates
    contour_path = os.path.join(sys_dir, 'planet_map_contours.txt')
    with open(contour_path, 'w') as f:
        f.write(f"# Contours for {sys_ind}\n")
        f.write(f"# Planet map contours\n")
        f.write(f"# Parameters: s2={s2}, q2={q2}, s3={s3}, q3={q3}, psi={psi}, rho={rho}, splanet={splanet}, qplanet={qplanet}, psi_planet={psi_planet}\n")
        f.write(f"# Planet rectangle corners: {planet_corners}\n")
        f.write("# Segment_ID, X, Y\n")
        for i, segment in enumerate(cp.allsegs[0]):
            for point in segment:
                f.write(f"{i}, {point[0]:.6f}, {point[1]:.6f}\n")
    

    return bin_u0, planet_corners

def calc_detectability(bin_u0, planet_corners, res, sys_ind, outdir, alpha_density=10.0):
    """
    Code to calculate the fraction of trajectories that will have both binary and planetary perturbations greater than a threshold
    bin_u0: Radius of circle around the binary caustic with fractional deviation from point lens greater than threshold
    planet_corners: corners of the rectangle around the planet caustic with fractional deviation from point lens greater than threshold
    res: Resolution of the magnification map
    sys_ind: Folder name of the system
    outdir: Output directory
    alpha_density: Number of alpha values inside the angle subtended by the planet rectangle at the origin

    Returns:
        n_det: Number of detectable trajectories
        n_u: Number of u values
        n_alpha: Number of alpha values
    """

    #Get parameters of the system from planet_map.fits
    with fits.open(os.path.join(outdir, sys_ind, 'planet_map.fits')) as hdul:
        header = hdul[1].header
        s2 = header['s2']
        q2 = header['q2']
        s3 = header['s3']
        q3 = header['q3']
        psi = header['psi']
        rho = header['rho']
        splanet = header['splanet']
        qplanet = header['qplanet']
        psi_planet = header['psi_plan']

    psi_planet = np.radians(psi_planet)

    #Rotate the planet rectangle corners by psi_planet degrees counterclockwise while maintaining their distances from the origin
    planet_corners_rotated = []
    for corner in planet_corners:
        x = corner[0]
        y = corner[1]
        x_rot = x*np.cos(psi_planet) - y*np.sin(psi_planet)
        y_rot = x*np.sin(psi_planet) + y*np.cos(psi_planet)
        planet_corners_rotated.append((x_rot, y_rot))
    
    #Create u list
    u = np.arange(-bin_u0, bin_u0, res)

    #Angle subtended by the planet rectangle at the origin
    alpha_p1 = np.arctan2(planet_corners_rotated[1][1], planet_corners_rotated[1][0])
    alpha_p2 = np.arctan2(planet_corners_rotated[2][1], planet_corners_rotated[2][0])
    alpha_pc1 = np.abs(alpha_p1 - alpha_p2)
    if alpha_p1 < 0:
        alpha_p1 = 2*np.pi + alpha_p1
    if alpha_p2 < 0:
        alpha_p2 = 2*np.pi + alpha_p2
    alpha_pc2 = np.abs(alpha_p1 - alpha_p2)
    alpha_planet_corners = np.min([alpha_pc1, alpha_pc2])
    alpha_inc = alpha_planet_corners/alpha_density

    #Create alpha list
    alpha = np.arange(0.0, np.pi, alpha_inc)

    print(f"Calculating detectability with {len(u)} u values and {len(alpha)} alpha values")


    det_traj = []
    for u0 in u:
        for alpha0 in alpha:
            #Check if the line y = tan(alpha0)x + u0/cos(alpha0) passes through the planet rectangle
            intersect = []
            for point in planet_corners_rotated:
                intersect.append(np.sign(np.tan(alpha0)*point[0] + u0/np.cos(alpha0) - point[1]))
            intersect = np.array(intersect)
            if np.all(intersect > 0) or np.all(intersect < 0):
                continue
            else:
                #count number of zeros in intersect
                num_zeros = np.sum(intersect == 0)
                if num_zeros != 1:
                    det_traj.append((u0, alpha0))
    
    #Write det_traj to a text file
    with open(os.path.join(outdir, sys_ind, 'det_traj.txt'), 'w') as f:
        f.write(f"# u0  alpha(deg)\n")
        for traj in det_traj:
            f.write(f"{traj[0]:.4f} {np.degrees(traj[1]):.2f}\n")

    return len(det_traj), len(u), len(alpha)

def calc_lc(u0, alpha, outdir, sys_ind, planet = True, bin = True):
    """
    Function to calculate light curves using binary and planet magnification maps and the trajectory parameters.
    Returns the magnification values along the trajectory for both binary and planet maps.
    
    u0: distance of the trajectory from the origin
    alpha: angle of the trajectory relative to the x-axis (in radians)
    outdir: Output directory
    sys_ind: Folder name of the system
    planet: Boolean to indicate if planet map should be used
    Returns:
        bin_tau: array of tau (time) values for binary map
        bin_mag: array of magnification values along trajectory in binary map
        planet_tau: array of tau (time) values for planet map
        planet_mag: array of magnification values along trajectory in planet map
    """
    # Equation of trajectory: y = tan(alpha)*x + u0/cos(alpha)
    # Or in parametric form: x = -u0*sin(alpha) + tau*cos(alpha), y = u0*cos(alpha) + tau*sin(alpha)
    # where tau is the time parameter (distance along trajectory from closest approach)
    
    # Get binary magnification map and metadata
    with fits.open(os.path.join(outdir, sys_ind, 'bin_map.fits')) as hdul:
        bin_map = hdul[1].data
        bin_header = hdul[1].header
        bin_box = bin_header['bin_box']
    
    # Get planet magnification map and metadata
    with fits.open(os.path.join(outdir, sys_ind, 'planet_map.fits')) as hdul:
        planet_map = hdul[1].data
        planet_header = hdul[1].header
        tau1 = planet_header['tau1']
        tau2 = planet_header['tau2']
        psi_planet = np.radians(planet_header['psi_plan'])
        planet_box = planet_header['planet_b']  # FITS header key is truncated
    
    # Define coordinate grids for each map
    # Binary map is centered at origin with extent [-bin_box/2, bin_box/2]
    bin_x = np.linspace(-bin_box/2, bin_box/2, bin_map.shape[1])
    bin_y = np.linspace(-bin_box/2, bin_box/2, bin_map.shape[0])
    bin_dx = bin_x[1] - bin_x[0]
    
    # Planet map has extent [tau1, tau2] in x and [-planet_box/2, planet_box/2] in y
    planet_x = np.linspace(tau1, tau2, planet_map.shape[1])
    planet_y = np.linspace(-planet_box/2, planet_box/2, planet_map.shape[0])
    
    if bin:
    # --- Extract magnification along trajectory for BINARY map ---
    # The trajectory passes through the binary map centered at origin
    # Find the range of tau where the trajectory intersects the binary map box
    
    # Parametric form: x = -u0*sin(alpha) + tau*cos(alpha), y = u0*cos(alpha) + tau*sin(alpha)
    # The line intersects the box when:
    #   -bin_box/2 <= -u0*sin(alpha) + tau*cos(alpha) <= bin_box/2  AND
    #   -bin_box/2 <= u0*cos(alpha) + tau*sin(alpha) <= bin_box/2
    
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Find tau range from x constraints: -bin_box/2 <= -u0*sin(alpha) + tau*cos(alpha) <= bin_box/2
        if np.abs(cos_alpha) > 1e-10:
            tau_x_min = (-bin_box/2 + u0*sin_alpha) / cos_alpha
            tau_x_max = (bin_box/2 + u0*sin_alpha) / cos_alpha
            if tau_x_min > tau_x_max:
                tau_x_min, tau_x_max = tau_x_max, tau_x_min
        else:
            # Vertical trajectory - check if x = -u0*sin(alpha) is within the box
            x_const = -u0 * sin_alpha
            if np.abs(x_const) > bin_box/2:
                tau_x_min, tau_x_max = np.inf, -np.inf  # No intersection
            else:
                tau_x_min, tau_x_max = -np.inf, np.inf
        
        # Find tau range from y constraints: -bin_box/2 <= u0*cos(alpha) + tau*sin(alpha) <= bin_box/2
        if np.abs(sin_alpha) > 1e-10:
            tau_y_min = (-bin_box/2 - u0*cos_alpha) / sin_alpha
            tau_y_max = (bin_box/2 - u0*cos_alpha) / sin_alpha
            if tau_y_min > tau_y_max:
                tau_y_min, tau_y_max = tau_y_max, tau_y_min
        else:
            # Horizontal trajectory - check if y = u0*cos(alpha) is within the box
            y_const = u0 * cos_alpha
            if np.abs(y_const) > bin_box/2:
                tau_y_min, tau_y_max = np.inf, -np.inf  # No intersection
            else:
                tau_y_min, tau_y_max = -np.inf, np.inf
        
        # Intersection of constraints
        bin_tau_min = max(tau_x_min, tau_y_min)
        bin_tau_max = min(tau_x_max, tau_y_max)
        
        if bin_tau_min < bin_tau_max:
            # Sample along the trajectory at the grid resolution
            num_samples = int((bin_tau_max - bin_tau_min) / bin_dx) + 1
            bin_tau = np.linspace(bin_tau_min, bin_tau_max, num_samples)
            
            # Calculate x, y positions along trajectory
            bin_traj_x = -u0*sin_alpha + bin_tau * cos_alpha
            bin_traj_y = u0*cos_alpha + bin_tau * sin_alpha
            
            # Convert to grid indices
            bin_dy = bin_y[1] - bin_y[0]
            bin_j = np.round((bin_traj_x - bin_x[0]) / bin_dx).astype(int)
            bin_i = np.round((bin_traj_y - bin_y[0]) / bin_dy).astype(int)
            
            # Clip to valid range
            valid_mask = (bin_j >= 0) & (bin_j < bin_map.shape[1]) & (bin_i >= 0) & (bin_i < bin_map.shape[0])
            bin_tau = bin_tau[valid_mask]
            bin_i = bin_i[valid_mask]
            bin_j = bin_j[valid_mask]
            
            # Extract magnification values
            bin_mag = bin_map[bin_i, bin_j]
        else:
            bin_tau = np.array([])
            bin_mag = np.array([])
    else:
        bin_tau = np.array([])
        bin_mag = np.array([])
        
    if planet:
        # --- Extract magnification along trajectory for PLANET map ---
        # The planet map is rotated by psi_planet, so we need to rotate the trajectory
        # to the planet's coordinate system
        
        # Rotate trajectory direction by -psi_planet to get into planet frame
        alpha_planet = alpha - psi_planet
        cos_alpha_p = np.cos(alpha_planet)
        sin_alpha_p = np.sin(alpha_planet)
        
        # The impact parameter u0 stays the same (it's the perpendicular distance)
        # In the rotated frame, the trajectory is: 
        #   x' = -u0*sin(alpha') + tau*cos(alpha'), y' = u0*cos(alpha') + tau*sin(alpha')
        
        planet_dx = planet_x[1] - planet_x[0]
        planet_dy = planet_y[1] - planet_y[0]
        
        # Find tau range from x constraints (planet map x range is [tau1, tau2])
        # tau1 <= -u0*sin(alpha') + tau*cos(alpha') <= tau2
        if np.abs(cos_alpha_p) > 1e-10:
            tau_x_min = (tau1 + u0*sin_alpha_p) / cos_alpha_p
            tau_x_max = (tau2 + u0*sin_alpha_p) / cos_alpha_p
            if tau_x_min > tau_x_max:
                tau_x_min, tau_x_max = tau_x_max, tau_x_min
        else:
            # Vertical trajectory - check if x = -u0*sin(alpha') is within [tau1, tau2]
            x_const = -u0 * sin_alpha_p
            if x_const < tau1 or x_const > tau2:
                tau_x_min, tau_x_max = np.inf, -np.inf  # No intersection
            else:
                tau_x_min, tau_x_max = -np.inf, np.inf
        
        # Find tau range from y constraints
        # planet_y[0] <= u0*cos(alpha') + tau*sin(alpha') <= planet_y[-1]
        if np.abs(sin_alpha_p) > 1e-10:
            tau_y_min = (planet_y[0] - u0*cos_alpha_p) / sin_alpha_p
            tau_y_max = (planet_y[-1] - u0*cos_alpha_p) / sin_alpha_p
            if tau_y_min > tau_y_max:
                tau_y_min, tau_y_max = tau_y_max, tau_y_min
        else:
            # Horizontal trajectory - check if y = u0*cos(alpha') is within the box
            y_const = u0 * cos_alpha_p
            if y_const < planet_y[0] or y_const > planet_y[-1]:
                tau_y_min, tau_y_max = np.inf, -np.inf  # No intersection
            else:
                tau_y_min, tau_y_max = -np.inf, np.inf
        
        # Intersection of constraints
        planet_tau_min = max(tau_x_min, tau_y_min)
        planet_tau_max = min(tau_x_max, tau_y_max)
        
        if planet_tau_min < planet_tau_max:
            # Sample along the trajectory
            num_samples = int((planet_tau_max - planet_tau_min) / planet_dx) + 1
            planet_tau = np.linspace(planet_tau_min, planet_tau_max, num_samples)
            
            # Calculate x, y positions along trajectory in planet frame
            planet_traj_x = -u0*sin_alpha_p + planet_tau * cos_alpha_p
            planet_traj_y = u0*cos_alpha_p + planet_tau * sin_alpha_p
            
            # Convert to grid indices
            planet_j = np.round((planet_traj_x - planet_x[0]) / planet_dx).astype(int)
            planet_i = np.round((planet_traj_y - planet_y[0]) / planet_dy).astype(int)
            
            # Clip to valid range
            valid_mask = (planet_j >= 0) & (planet_j < planet_map.shape[1]) & (planet_i >= 0) & (planet_i < planet_map.shape[0])
            planet_tau = planet_tau[valid_mask]
            planet_i = planet_i[valid_mask]
            planet_j = planet_j[valid_mask]
            
            # Extract magnification values
            planet_mag = planet_map[planet_i, planet_j]
        else:
            planet_tau = np.array([])
            planet_mag = np.array([])
    
    else:
        planet_tau = np.array([])
        planet_mag = np.array([])
        
    return bin_tau, bin_mag, planet_tau, planet_mag


def _process_trajectory(u0, alpha, outdir, sys_ind, threshold, type):
    """
    Helper function to process a single trajectory for parallel execution.
    
    Args:
        u0: Impact parameter
        alpha: Trajectory angle in degrees
        outdir: Output directory
        sys_ind: System identifier
        threshold: Threshold for deviation
        type: Type of caustic region ('both', 'bin', or 'planet')
    
    Returns:
        tuple: (u0, alpha, n_points_bin, n_points_planet)
    """
    if type == 'both':
        bin_tau, bin_mag, planet_tau, planet_mag = calc_lc(u0, np.radians(alpha), outdir, sys_ind)
    elif type == 'bin':
        bin_tau, bin_mag, planet_tau, planet_mag = calc_lc(u0, np.radians(alpha), outdir, sys_ind, bin=True, planet=False)
    elif type == 'planet':
        bin_tau, bin_mag, planet_tau, planet_mag = calc_lc(u0, np.radians(alpha), outdir, sys_ind, bin=False, planet=True)
    
    # Count consecutive points above threshold for binary
    n_points_bin = 0
    if len(bin_mag) > 0:
        bin_mag_abs = np.abs(bin_mag)
        mask = bin_mag_abs > threshold
        max_seq = 0
        curr_seq = 0
        for val in mask:
            if val:
                curr_seq += 1
                if curr_seq > max_seq:
                    max_seq = curr_seq
            else:
                curr_seq = 0
        n_points_bin = max_seq
    
    # Count consecutive points above threshold for planet
    n_points_planet = 0
    if len(planet_mag) > 0:
        planet_mag_abs = np.abs(planet_mag)
        mask = planet_mag_abs > threshold
        max_seq = 0
        curr_seq = 0
        for val in mask:
            if val:
                curr_seq += 1
                if curr_seq > max_seq:
                    max_seq = curr_seq
            else:
                curr_seq = 0
        n_points_planet = max_seq
    
    return (u0, alpha, n_points_bin, n_points_planet)

    
def refine_det(outdir, sys_ind, threshold=0.05, n_pot = 3, type = 'both', num_cores=1):
    """
    Function to refine detectability estimates by creating light curves and checking for deviations above a threshold (parallelized)
    outdir: Output directory
    sys_ind: Folder name of the system
    threshold: Threshold for deviation
    n_pot: Number of points above threshold to be considered detectable
    type: Type of caustic region ('both' for circumbinary, 'bin' for binary only, 'planet' for planet only)
    num_cores: Number of CPU cores to use for parallel processing
    Returns:
        n_det: Number of detectable trajectories
    """
    if type == 'both':
        #Load u0 and alpha values from det_traj.txt
        with open(os.path.join(outdir, sys_ind, 'det_traj.txt'), 'r') as f:
            det_traj = f.readlines()
    elif type == 'bin':
        #Load u0 and alpha values from det_traj_bin.txt
        with open(os.path.join(outdir, sys_ind, 'det_traj_bin.txt'), 'r') as f:
            det_traj = f.readlines()
    elif type == 'planet':
        #Load u0 and alpha values from det_traj_planet.txt
        with open(os.path.join(outdir, sys_ind, 'det_traj_planet.txt'), 'r') as f:
            det_traj = f.readlines()
    else:
        raise ValueError("Invalid type. Must be 'both', 'bin', or 'planet'")
    #skip lines starting with #
    det_traj = [line for line in det_traj if not line.startswith('#')]
    det_traj = [line.split() for line in det_traj]
    det_traj = [(float(line[0]), float(line[1])) for line in det_traj]
    
    # Process trajectories (parallel or sequential depending on num_cores)
    if num_cores == 1:
        # Sequential processing - no multiprocessing overhead
        results = []
        for u0, alpha in det_traj:
            results.append(_process_trajectory(u0, alpha, outdir, sys_ind, threshold, type))
        n_points_bin = [r[2] for r in results]
        n_points_planet = [r[3] for r in results]
    else:
        # Parallel processing
        results = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_trajectory, u0, alpha, outdir, sys_ind, threshold, type): (u0, alpha) 
                       for u0, alpha in det_traj}
            
            # Collect results as they complete
            for future in as_completed(futures):
                results.append(future.result())
        
        # Sort results by original order (u0, alpha)
        results_dict = {(r[0], r[1]): (r[2], r[3]) for r in results}
        n_points_bin = [results_dict[(u0, alpha)][0] for u0, alpha in det_traj]
        n_points_planet = [results_dict[(u0, alpha)][1] for u0, alpha in det_traj]
    n_det = 0
    if type == 'both':
        for i in range(len(n_points_bin)):
            if n_points_bin[i] >= n_pot and n_points_planet[i] >= n_pot:
                n_det += 1
        u0 = np.array([traj[0] for traj in det_traj])
        alpha = np.array([traj[1] for traj in det_traj])
        n_points_bin = np.array(n_points_bin)
        n_points_planet = np.array(n_points_planet)
        np.savetxt(os.path.join(outdir, sys_ind, 'det_traj.txt'), np.array([u0, alpha, n_points_bin, n_points_planet]).T, fmt = '%.4f %.2f %.0f %.0f', header='u0  alpha(deg)  n_points_bin  n_points_planet')
    elif type == 'bin':
        for i in range(len(n_points_bin)):
            if n_points_bin[i] >= n_pot:
                n_det += 1
        u0 = np.array([traj[0] for traj in det_traj])
        alpha = np.array([traj[1] for traj in det_traj])
        n_points_bin = np.array(n_points_bin)
        np.savetxt(os.path.join(outdir, sys_ind, 'det_traj_bin.txt'), np.array([u0, alpha, n_points_bin]).T, fmt = '%.4f %.2f %.0f', header='u0  alpha(deg)  n_points_bin')
    elif type == 'planet':
        for i in range(len(n_points_planet)):
            if n_points_planet[i] >= n_pot:
                n_det += 1
        u0 = np.array([traj[0] for traj in det_traj])
        alpha = np.array([traj[1] for traj in det_traj])
        n_points_planet = np.array(n_points_planet)
        np.savetxt(os.path.join(outdir, sys_ind, 'det_traj_planet.txt'), np.array([u0, alpha, n_points_planet]).T, fmt = '%.4f %.2f %.0f', header='u0  alpha(deg)  n_points_planet')

    return n_det

    



     