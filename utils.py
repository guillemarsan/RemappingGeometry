import numpy as np
import scipy

def compute_meshgrid(radius, num_bins):
    radius = 1
    row_col = int(np.sqrt(num_bins))
    step = 2*radius/row_col
    
    rows = np.arange(-radius+step/2, radius, step=step)
    ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T

    return ppts

def compute_ratemap(path, spikes, dt, bins):

    num_bins = bins.shape[1]
    step = np.abs(np.sum(bins[:,0]-bins[:,1])/2)
    sums = np.zeros(num_bins)
    for j in np.arange(num_bins):
        dist = np.max(np.abs(np.expand_dims(bins[:,j],-1)-path), axis=0) # Manhattan distance
        consider = np.argwhere(dist < step)
        sums[j] = np.sum(spikes[consider])/(consider.size*dt) if consider.size != 0 else 0
    return sums

def compute_pf(ratemap, bins):
    rows = int(np.sqrt(bins.shape[1]))

    ratemap_square = np.reshape(ratemap, (rows,rows))
    smoothed = scipy.ndimage.gaussian_filter(ratemap_square, sigma=1)

    maxfr = np.max(smoothed)
    smoothed_cutoff = smoothed.copy()
    smoothed_cutoff[smoothed < 0.2*maxfr] = 0
    

    minpf = 0.4 # Min pf size in m 0.15 => 225cm2
    side = round(np.abs(np.sum(bins[:,0]-bins[:,1])),7)
    minbins = int(minpf/side)
    structure = np.ones((3, 3), dtype=np.int)

    
    labeled, ncomponents = scipy.ndimage.measurements.label(smoothed_cutoff>0, structure)
    

    for com in np.arange(ncomponents):
        com_lab = np.argwhere(labeled == com)
        if com_lab.shape[0] > minbins:
            labeled[com_lab] = 1
        else: labeled[com_lab] = 0

    pf = smoothed_cutoff * (labeled>0)  
    pf = np.reshape(pf, (bins.shape[1],))
    return pf