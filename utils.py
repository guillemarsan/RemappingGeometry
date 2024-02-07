import numpy as np
import time
from scipy import ndimage 
import matplotlib.pyplot as plt

def compute_meshgrid(radius, num_bins, dim=2):

    row_col = int(np.sqrt(num_bins)) if dim == 2 else num_bins
    step = 2*radius/row_col
    
    rows = np.arange(-radius+step/2, radius, step=step)
    ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T if dim == 2 else rows[np.newaxis,:]

    return ppts

def compute_pathloc(path, bins, dim=2):
    num_bins = bins.shape[1]
    time_steps = path.shape[1]
    step = np.abs(np.sum(bins[:,0]-bins[:,1])/2)
    pathloc = np.zeros((num_bins,time_steps))

    if dim == 2:
        origin = np.array([-1,1])[:,np.newaxis]
        idcs = (np.abs(path - origin) // (2*step)).astype(int)
        j = idcs[0,:] + int(np.sqrt(num_bins))*idcs[1,:]
    else:
        origin = np.array([-1])[:,np.newaxis]
        idcs = (np.abs(path - origin) // (2*step)).astype(int)
        j = idcs[0,:]
    pathloc[j,np.arange(time_steps)] = 1

    return pathloc.astype(int)

def compute_ratemap_r(rates, pathloc):
    
    num_bins = pathloc.shape[0]
    ratemap = np.zeros(num_bins)
    for j in np.arange(num_bins):
        ratemap[j] = np.mean(rates[pathloc[j,:].astype(bool)]) if np.any(pathloc[j,:]) else 0
    return ratemap

def compute_ratemap_s(spikes, pathloc, tb):

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
    structure = np.ones((3, 3), dtype=int)

    
    labeled, ncomponents = ndimage.measurements.label(smoothed_cutoff>0, structure)
    

    for com in np.arange(ncomponents):
        com_lab = np.argwhere(labeled == com)
        if com_lab.shape[0] > minbins:
            labeled[com_lab] = 1
        else: labeled[com_lab] = 0

    pf = smoothed_cutoff * (labeled>0)  
    pf = np.reshape(pf, (bins.shape[1],))
    return pf

def check_code_encode(x, rt, D, T):
    DR = np.linalg.pinv(D)
    DTL = np.linalg.pinv(D.T)

    # check math
    dfdr = -2*D.T @ x + 2*D.T @ D @ rt + 2*T[:,np.newaxis]
    epsT = -DR @ DTL @ T[:,np.newaxis]
    epsr = 0.5* DR @ DTL @ dfdr
    eps = epsT + epsr
    print(np.max(np.abs(epsT)))
    print(np.max(np.abs(epsr)))
    print(np.max(np.abs(eps)))
    nu = rt - DR @ x - eps
    print(np.max(np.abs(nu)))
    print(np.max(np.abs(D @ nu)))

    # everything is correct
    rnew = DR @ x + eps + nu
    print(np.max(np.abs(rt - rnew)))

    # check r
    rhatT = rt - epsT
    rhatr = rt - epsr
    rhatTr = rt - epsT - epsr
    rhatTrnu = rt - epsT - epsr - nu

    # plt.figure()
    plt.plot(rt[7,:], label='r')
    plt.plot(rhatT[7,:], label='r - eT')
    plt.plot(rhatr[7,:], label='r - er')
    plt.plot(rhatTr[7,:], label='r - eT - er')
    plt.plot(rhatTrnu[7,:], label='r - eT - er  - nu = DRx')
    plt.legend()

    # check x
    xhat = D @ rt
    xhatT = xhat - D @ epsT
    xhatr = xhat - D @ epsr
    xhatTr = xhat - D @ epsT - D @ epsr
    print(np.max(np.abs(x - xhat)))
    print(np.max(np.abs(x - xhatT)))
    print(np.max(np.abs(x - xhatr)))
    print(np.max(np.abs(x - xhatTr)))

    plt.figure()
    plt.plot(x[1,:], label='x', color='k')
    plt.plot(xhat[1,:], label='Dr')
    plt.plot(xhatT[1,:], label='Dr - DeT')
    plt.plot(xhatr[1,:], label='Dr - Der')
    plt.plot(xhatTr[1,:], label='Dr - DeT - Der')
    plt.legend()

    return
