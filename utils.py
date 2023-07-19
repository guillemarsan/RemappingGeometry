import numpy as np
from scipy import ndimage 

def compute_meshgrid(radius, num_bins):

    row_col = int(np.sqrt(num_bins))
    step = 2*radius/row_col
    
    rows = np.arange(-radius+step/2, radius, step=step)
    ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T

    return ppts

def compute_pathloc(path, dt, time_steps, bins):
    num_bins = bins.shape[1]
    step = np.abs(np.sum(bins[:,0]-bins[:,1])/2)
    pathloc = np.zeros((num_bins,time_steps))
    for j in np.arange(num_bins):
        dist = np.max(np.abs(np.expand_dims(bins[:,j],-1)-path), axis=0) # Manhattan distance
        pathloc[j,np.argwhere(dist < step)] = 1
    return pathloc.astype(int)

def compute_ratemap(spikes, pathloc, tb):

    num_bins = tb.size
    ratemap = np.zeros(num_bins)
    for j in np.arange(num_bins):
        ratemap[j] = np.sum(spikes[pathloc[j,:].astype(bool)])/tb[j] if tb[j] != 0 else 0
    return ratemap

def compute_pf(ratemap, bins):
    rows = int(np.sqrt(bins.shape[1]))

    ratemap_square = np.reshape(ratemap, (rows,rows))
    smoothed = ndimage.gaussian_filter(ratemap_square, sigma=1)

    maxfr = np.max(smoothed)
    smoothed_cutoff = smoothed.copy()
    smoothed_cutoff[smoothed < 0.2*maxfr] = 0
    

    minpf = 0.4 # Min pf size in m 0.15 => 225cm2
    side = round(np.abs(np.sum(bins[:,0]-bins[:,1])),7)
    minbins = int(minpf/side)**2
    structure = np.ones((3, 3), dtype=np.int)

    
    labeled, ncomponents = ndimage.measurements.label(smoothed_cutoff>0, structure)
    

    for com in np.arange(ncomponents):
        com_lab = np.argwhere(labeled == com)
        if com_lab.shape[0] > minbins:
            labeled[com_lab] = 1
        else: labeled[com_lab] = 0

    pf = smoothed_cutoff * (labeled>0)  
    pf = np.reshape(pf, (bins.shape[1],))
    return pf