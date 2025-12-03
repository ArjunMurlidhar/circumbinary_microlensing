import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import glob
import os
import csv
import argparse

def process_map(filepath, output_dir, make_plots=False):
    filename = os.path.basename(filepath)
    file_root = filename.replace('.fits.fz', '').replace('.fits', '')
    
    try:
        with fits.open(filepath) as hdul:
            # Try to get header from the compressed extension (usually index 1) or primary
            if len(hdul) > 1:
                header = hdul[1].header
                data = hdul[1].data
            else:
                header = hdul[0].header
                data = hdul[0].data
            
            # Extract parameters
            side = header.get('SIDE', 2.0) # Default to 2.0 if missing (based on observation)
            naxis1 = header.get('NAXIS1', data.shape[1])
            res = side / naxis1
            
            # Create coordinate arrays (centered at 0)
            # Corresponds to np.linspace(-side/2, side/2, naxis)
            # Pixel 0 is at -side/2
            
            # 1. Calculate Area where abs(data) > 0.05
            abs_data = np.abs(data)
            mask = abs_data > 0.05
            pixel_count = np.sum(mask)
            physical_area = pixel_count * (res ** 2)
            
            # 2. Extract Contours
            
            # We contour on the physical grid extent to get physical coordinates directly
            x = np.linspace(-side/2, side/2, data.shape[1])
            y = np.linspace(-side/2, side/2, data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Use matplotlib to compute contour segments even if we don't save the plot
            # Using a figure is necessary to get the ContourSet object
            fig, ax = plt.subplots(figsize=(10, 10))
            
            if make_plots:
                # Plotting the map for background
                im = ax.imshow(abs_data, origin='lower', extent=[-side/2, side/2, -side/2, side/2], cmap='viridis')
                
                # Overlay shaded region for mask > 0.05
                # Method using contourf for filled regions (cleaner with contours)
                # Shade regions > 0.05 with semi-transparent white/cyan
                ax.contourf(X, Y, abs_data, levels=[0.05, np.max(abs_data)], colors='white', alpha=0.3)
            
            # Calculate contours (needed for both coords and optional plot)
            cs = ax.contour(X, Y, abs_data, levels=[0.05], colors='red', linewidths=1)
            
            if make_plots:
                # Save Plot
                ax.set_title(f"{filename}\nArea > 0.05: {physical_area:.4f} (E^2)")
                plt.colorbar(im, label='Abs(Magnification)')
                plot_path = os.path.join(output_dir, 'plots', f"{file_root}.png")
                plt.savefig(plot_path)
            
            plt.close(fig)
            
            # 3. Save Contour Coordinates
            contour_path = os.path.join(output_dir, 'contours', f"{file_root}.txt")
            with open(contour_path, 'w') as f:
                f.write(f"# Contours for {filename}\n")
                f.write(f"# Physical Area > 0.05: {physical_area}\n")
                f.write(f"# Pixel Count: {pixel_count}\n")
                f.write("# Segment_ID, X, Y\n")
                
                # cs.allsegs is a list of levels (we only have one level: 0.05)
                # each level is a list of segments (numpy arrays of shape (N, 2))
                for i, segment in enumerate(cs.allsegs[0]):
                    for point in segment:
                        f.write(f"{i}, {point[0]:.6f}, {point[1]:.6f}\n")
            
            return {
                'Filename': filename,
                'Pixel_Count': pixel_count,
                'Physical_Area': physical_area,
                'Side_Length': side,
                'Resolution': res
            }
            
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze magnification map contours.")
    parser.add_argument('--plot', action='store_true', help="Enable generation of contour plots.")
    args = parser.parse_args()

    input_dir = 'magnification_maps'
    output_dir = 'contour_analysis_results'
    csv_path = os.path.join(output_dir, 'summary.csv')
    files = sorted(glob.glob(os.path.join(input_dir, '*.fits.fz')))
    
    if not files:
        print("No .fits.fz files found in magnification_maps/")
        return

    print(f"Found {len(files)} files. Starting analysis...")
    
    #if csv file exists, read it
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            results = [row for row in reader]
    else:
        results = []
        
    for f in files:
        if os.path.basename(f) in [row['Filename'] for row in results]:
            print(f"Skipping {os.path.basename(f)} because it already exists in the summary.csv")
            continue
        print(f"Processing {os.path.basename(f)}...")
        res = process_map(f, output_dir, make_plots=args.plot)
        if res:
            results.append(res)
            
    # Save Summary CSV
    keys = ['Filename', 'Pixel_Count', 'Physical_Area', 'Side_Length', 'Resolution']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
