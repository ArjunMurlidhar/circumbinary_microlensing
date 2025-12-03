import numpy as np
import VBMicrolensing as vbm
import astropy.io.fits as fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def _compute_mag_row(args):
    """
    Compute one row (fixed u, varying t) of the magnification map.
    Returns a tuple (i, row) so the caller can place it correctly.
    """
    VBM = vbm.VBMicrolensing()
    VBM.Tol = 1e-4
    i, params, tau, ulist, use_float32 = args
    # Compute with float64 for stability, then cast if requested
    row64 = np.empty(tau.shape[0], dtype=np.float64)
    if len(params) == 1:
        rho = params['rho']
        for j in range(tau.shape[0]):
            u = np.sqrt(float(ulist[i])**2 + float(tau[j])**2)
            row64[j] = VBM.ESPLMag2(u , rho)

    elif len(params) == 3:
        #binary star system with axis along the x-axis
        s, q, rho = params['s'], params['q'], params['rho']
        u = float(ulist[i])
        for j in range(tau.shape[0]):
            row64[j] = VBM.BinaryMag2(s, q, float(tau[j]), u, rho)

    elif len(params) == 4:
        #Planet with star at the origin 
        s, q, psi, rho = params['s'], params['q'], params['psi'], params['rho']
        psi = np.deg2rad(psi)
        # Masses normalized so that total mass = 1
        m1 = 1.0 / (1.0 + q)
        m2 = q * m1
        

        # Lens positions (binary along x, planet offset by s3 at angle psi)
        z1x = 0.0
        z1y = 0.0
        z3x = s * np.cos(psi)
        z3y = s * np.sin(psi)

        vbm_params = [
            z1x, z1y, m1,
            z3x, z3y, m2
        ]
        VBM.SetMethod(VBM.Method.Multipoly)
        VBM.SetLensGeometry(vbm_params)
        
        u = float(ulist[i])
        for j in range(tau.shape[0]):
            row64[j] = VBM.MultiMag(float(tau[j]), u , rho)
    elif len(params) == 6:
        #Circumbinary system
        s2, q2, rho, s3, q3, psi = params['s2'], params['q2'], params['rho'], params['s3'], params['q3'], params['psi']
        psi = np.deg2rad(psi)
        # Masses normalized so that total binary mass = 1
        m1 = 1.0 / (1.0 + q2)
        m2 = q2 * m1
        m3 = q3 * m1

        # Lens positions (binary along x, planet offset by s3 at angle psi)
        z1x = -q2 * s2 / (1.0 + q2)
        z2x = s2 / (1.0 + q2)
        z3x = s3 * np.cos(psi) 
        z3y = s3 * np.sin(psi)

        vbm_params = [
        z1x, 0.0, m1,  # lens 1 (x, y, m)
        z2x, 0.0, m2,  # lens 2
        z3x, z3y, m3  # lens 3 (planet)
        ]
        VBM.SetMethod(VBM.Method.Multipoly)
        VBM.SetLensGeometry(vbm_params)
        u = float(ulist[i])
        for j in range(tau.shape[0]):
            row64[j] = VBM.MultiMag(float(tau[j]), u , rho)
    else:
        raise ValueError("Invalid number of parameters")

    if use_float32:
        return i, row64.astype(np.float32, copy=False)
    return i, row64


def create_mag_map(params, res=1e-3, side=8, box_dim = None, num_cores=1, float_precision='float64', backend='auto'):
    '''
    Create a magnification map for a given binary or circumbinary lens model.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the microlensing model (with finite source size).
    res : float, optional
        Resolution of the grid in units of the Einstein radius.
    side : float, optional
        Side length of the grid in units of the Einstein radius.
    box_dim : list, optional
        [(tau1, tau2), (u1, u2)]: Box dimensions in units of the Einstein radius for the tau and u axes respectively. If None, side is used.
    num_cores : int, optional
        Number of CPU cores to use for parallel computation (default 1 = serial).
    float_precision : {'float32', 'float64'}, optional
        Precision of the returned magnification map. Calculations are performed
        in float64 and cast to float32 if requested.
    backend : {'auto', 'processes', 'threads'}, optional
        Parallel backend. Use 'threads' in notebooks if processes cause errors.

    Returns
    -------
    mag_map : numpy.ndarray
        A 2D array representing the magnification map, with the requested dtype.
    '''
    if box_dim is None:
        # Grid setup
        t_steps = int(side / res)
        u_steps = int(side / res)
        # Axes (computed in float64 regardless of output dtype)
        tau = np.linspace(-side/2, side/2, t_steps, dtype=np.float64)
        ulist = np.linspace(-side/2, side/2, u_steps, dtype=np.float64)
    else:
        t_steps = int((box_dim[0][1] - box_dim[0][0]) / res)
        u_steps = int((box_dim[1][1] - box_dim[1][0]) / res)
        tau = np.linspace(box_dim[0][0], box_dim[0][1], t_steps, dtype=np.float64)
        ulist = np.linspace(box_dim[1][0], box_dim[1][1], u_steps, dtype=np.float64)

    use_float32 = str(float_precision).lower() == 'float32'
    out_dtype = np.float32 if use_float32 else np.float64

    # Shape (u_steps, t_steps): each row corresponds to a u value, columns to t
    mag_map = np.empty((u_steps, t_steps), dtype=out_dtype)

    # Serial path
    if num_cores is None or int(num_cores) <= 1:
        for i in range(u_steps):
            _, row = _compute_mag_row((i, params, tau, ulist, use_float32))
            mag_map[i, :] = row
        return mag_map

    # Parallel path
    max_workers = min(int(num_cores), u_steps) if num_cores is not None else u_steps

    def _run_parallel_pool(executor_cls):
        with executor_cls(max_workers=max_workers) as ex:
            futures = [ex.submit(_compute_mag_row, (i, params, tau, ulist, use_float32)) for i in range(u_steps)]
            for fut in as_completed(futures):
                i, row = fut.result()
                mag_map[i, :] = row

    if backend == 'threads':
        _run_parallel_pool(ThreadPoolExecutor)
    elif backend == 'processes':
        _run_parallel_pool(ProcessPoolExecutor)
    else:  # auto
        try:
            _run_parallel_pool(ProcessPoolExecutor)
        except Exception:
            _run_parallel_pool(ThreadPoolExecutor)

    return mag_map


def save_mag_map(mag_map, filename,
                                 compression=None,
                                 tile_shape=None,
                                 quantize_level=None,
                                 metadata=None):
    '''
    Save magnification map as a compressed FITS file using tiled image compression.

    Parameters
    ----------
    mag_map : np.ndarray
        2D array with rows=u and columns=t.
    filename : str
        Output filename, e.g. 'mag_map.fits.fz' or 'mag_map.fits'.
    compression : {'GZIP_2', 'RICE_1', 'PLIO_1', 'HCOMPRESS_1'}
        Compression algorithm. Use 'GZIP_2' for lossless floats.
        Use 'RICE_1' with quantize_level for smaller, lossy float files.
    tile_shape : tuple[int, int] | None
        Tiling for compression. Defaults to up to 512x512.
    quantize_level : float | None
        Quantization step for floating point when using 'RICE_1'.
        Smaller values => better fidelity, larger => smaller files.
        Ignored for lossless (e.g., 'GZIP_2').
    metadata : dict | None
        Optional FITS header keywords to record (keys will be uppercased and
        truncated to 8 chars as needed).
    '''
    # Create HDU
    if compression is None:
        hdu = fits.ImageHDU(data=mag_map)
    else:
        if tile_shape is None:
            tile_shape = (min(mag_map.shape[0], 512), min(mag_map.shape[1], 512))
        
        hdu = fits.CompImageHDU(
            data=mag_map,
            compression_type=compression,
            tile_shape=tile_shape,
            quantize_level=quantize_level
        )

    # Add optional metadata
    if metadata:
        for k, v in metadata.items():
            key = str(k).upper()[:8]
            try:
                hdu.header[key] = v
            except Exception:
                # Skip values that cannot be serialized into FITS header
                pass

    # Write file (PrimaryHDU + compressed image extension)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(filename, overwrite=True)


# Example usage:
# meta = {'S': s, 'Q': q, 'RHO': rho, 'RES': res, 'SIDE': side, 'DTYPE': str(mag_map.dtype)}
# save_mag_map_compressed_fits(
#     mag_map,
#     filename='mag_map.fits.fz',
#     compression='GZIP_2',        # lossless for floats
#     tile_shape=None,             # defaults to up to 512x512 tiles
#     quantize_level=None,         # not used for GZIP_2
#     metadata=meta
# )
#
# For smaller (lossy) files with very good fidelity on float data:
# save_mag_map_compressed_fits(
#     mag_map.astype(np.float32, copy=False),
#     filename='mag_map_rice.fits.fz',
#     compression='RICE_1',
#     tile_shape=None,
#     quantize_level=16.0,         # adjust based on acceptable quantization
#     metadata=meta
# )

if __name__ == "__main__":
    magmap = create_mag_map(1.35, 0.0058, 0.0098, res=1e-3, side=8, num_cores=4, float_precision='float32', backend='threads')
    save_mag_map(magmap, 'magmap_test.fits.fz', compression='GZIP_2', tile_shape=None, quantize_level=None, metadata={'S': 1.35, 'Q': 0.0058, 'RHO': 0.0098, 'RES': 1e-3, 'SIDE': 8, 'DTYPE': str(magmap.dtype)})

