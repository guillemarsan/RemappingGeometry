from operator import ne
import pathlib, argparse, json, time
from utils import compute_meshgrid, compute_ratemap, compute_pf, compute_pathloc
import numpy as np
import time
import pandas as pd
from convexsnn.path import get_path

##################### STANDARDS ###################################

b = 100 #number of bins
radius = 1 #radius of the environment
bin_size = (2*radius)**2/b #bin_size
tostore = lambda array: np.array2string(array, separator=',', suppress_small=True, max_line_width=np.inf)

##################### DATA LOAD ###################################
def read_matrix(path,f,matrix='s'):
    patt = f[matrix]
    m_file = path.rglob('%s' % patt)
    f = next(m_file)
    t0 = time.time()
    m = np.genfromtxt(f)
    t1 = time.time()
    print(t1-t0)
    return m

def load_dataframe(keyname, basepath):

    patt = "*-%s_df.csv" % keyname
    path = pathlib.Path(basepath)
    results_files = path.rglob(patt)
    no_load = False
    try:
        file = next(results_files)
    except StopIteration:
        no_load = True

    if no_load:
        df = pd.DataFrame()
    else:
        old_file_name = str(file)
        with open(old_file_name, 'rb') as f:
            df = pd.read_csv(f, index_col=[0])

    return df

def read_pfs(set, neu, b, dirs):

    all_pfs = np.zeros((neu, b, dirs))
    i = 0
    for file in set['pfs']:
        with open(file) as res_file:
            t0 = time.time()
            pfs = np.genfromtxt(res_file)
            t1 = time.time()
            print(t1-t0)
        all_pfs[:,:,i] = pfs
        i += 1

    return all_pfs


##################### COMPUTES ####################################

def compute_database(basepath):

    patt = "*.json"
    path = pathlib.Path(basepath)
    results_files = path.rglob(patt)

    newdata = []
    for f in results_files:
        with open(f) as res_file:
            f = json.load(res_file)

            # Unroll args
            for args, value in f['args'].items():
                f['arg_'+ args] = value
            del f['args']

            delete_list = ['arg_dir', 'arg_plot', 'arg_gif', 'arg_save']
            for d in delete_list:
                del f[d]

            newdata.append(f)

    return pd.DataFrame(newdata)

def compute_per_simulation(dbase, params, func):
    return dbase.apply(lambda x: func(x, params), axis=1)

def compute_across(dbase, across, params, func):

    groupby_pars = params
    groupby_pars.remove(across)

    gb = dbase.groupby(groupby_pars, as_index=False)
    df = gb.apply(lambda x: func(x, groupby_pars))
    return df


##################### ANALYSE RESULTS ###################################

########## ADD TO MAIN DATABASE #############

def analyse_ratemaps_pfs(point):

    #TODO Eliminate stimes, Th, etc. tags from original 
    id = point['basepath']
    stimes_name = "{0}-stimes.csv".format(id)
    with open(stimes_name) as res_file:
        t0 = time.time()
        stimes = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    # Transform from stimes to spike raster
    stimes = stimes.astype(int)
    s = np.zeros((point['arg_nb_neurons'],point['nb_steps']))
    s[stimes[:,0],stimes[:,1]] = 1

    # TODO
    # p_seed = set['p_seed']
    # p_type = set['p_type']
    p_type = 'uspiral'
    p_seed = 0

    p, dp, t, dt, time_steps = get_path(dpcs=2, type=p_type, path_seed=p_seed)
    bins = compute_meshgrid(radius,b)
    neu = point['arg_nb_neurons']

    # Compute occupancies    
    pathloc = compute_pathloc(p, dt, time_steps, bins)
    tb = np.sum(pathloc, axis=1)*dt
    occupancies = tb/np.max(t)

    ratemaps = np.zeros((neu,b))
    placefields = np.zeros((neu,b))
    meanfr = np.zeros(neu)
    for i in np.arange(neu):
        ratemaps[i,:] = compute_ratemap(s[i,:], pathloc, tb)
        placefields[i,:] = compute_pf(ratemaps[i,:], bins)
        meanfr[i] = np.sum(occupancies*placefields[i,:])

    ratemaps_name = "{0}-ratemaps.csv".format(id)
    pfs_name = "{0}-pfs.csv".format(id)

    np.savetxt(ratemaps_name, ratemaps, fmt='%.3e')
    np.savetxt(pfs_name, placefields, fmt='%.3e')

    point['meanfr'] = tostore(meanfr)
    point['occupancies'] = tostore(occupancies)
    point['ratemaps'] = ratemaps_name
    point['pfs'] = pfs_name

    return point

def analyse_classes(set):

    neu = set.iloc[0]['arg_nb_neurons']
    set.sort_values(by=['arg_current_amp'])
    inhibis_vect = set['arg_current_amp']
    
    classes = np.ones(neu)*3
    original = set.iloc[0]
    active = np.array(eval(original['pcsidx']))
    # TODO the tagged
    # tagged = np.array(eval(original['taggedidx']))
    tagged = np.arange(20)*2
    classes[active] = 0
    classes[tagged] = 1

    set['classes'] = tostore(classes)

    i = 1
    for inhib in inhibis_vect[1:]:
        newclasses = classes

        casenow = set['arg_current_amp' == inhib]
        activenow = np.array(eval(casenow['pcsidx']))

        recruited = np.setdiff1d(activenow, active)

        classes[recruited] = 2

        casenow['classes'] = tostore(classes)
        i += 1

    return set

#########  PF ANALYSIS ##############

def analyse_placecells(row, params):
    point = row[params]
    analyse_spatialinfo(row, point)
    return point

def analyse_spatialinfo(row, point):

    with open(row['pfs']) as res_file:
        t0 = time.time()
        pfs = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    occupancies = np.array(eval(row['occupancies']))
    meanfr = np.array(eval(row['meanfr']))
    
    neu = row['arg_nb_neurons']
    
    # Compute information rate
    info_bin = np.zeros((b, neu))
    results_spatialinfo = np.zeros((neu, 1))
    for j in np.arange(b):
        active = np.where(pfs[:,j]>0)[0]
        if active.size != 0:
            info_bin[j,active] = occupancies[j]*pfs[active,j]*np.log2(pfs[active,j]/meanfr[active])
    results_spatialinfo = np.sum(info_bin, axis=0)

    # Histogram 
    info_bins = np.arange(0+1e-5,np.max(results_spatialinfo),9)
    info_bins = np.insert(info_bins,0,0)
    
    active_neurons = np.where(np.sum(pfs,axis=1))[0]
    hist = np.apply_along_axis(lambda a: np.histogram(a, bins=info_bins), 0, results_spatialinfo[active_neurons])[0]

    point['spatialinfo'] = tostore(hist)
    point['spatialinfo_bins'] = tostore(np.array(info_bins))

    return point



####### REMAPPING ANALYSIS ###########

def analyse_remapping(set, params):

    point = set.iloc[0][params]
    analyse_nrooms(set, point)
    analyse_overlap(set, point)
    
    neu = set.iloc[0]['arg_nb_neurons']
    dirs = len(set)
    all_pfs = read_pfs(set, neu, b, dirs)

    analyse_spatialcorr(point, all_pfs)
    analyse_multispatialcorr(point, all_pfs)
    return point

def analyse_nrooms(set, point):

    neus = set.iloc[0]['arg_nb_neurons']
    dirs = set['arg_input_dir'].shape[0]

    nbins=np.arange(dirs+2)
    results_nrooms = np.zeros(int(neus))
    for pcsidx in set['pcsidx']:
        results_nrooms[np.array(eval(pcsidx))] += 1 

    hist = np.histogram(results_nrooms, bins=nbins)[0].astype(float)
    
    point['nrooms'] = tostore(hist)
    point['nrooms_bins'] = tostore(np.arange(dirs+1)) 

def analyse_overlap(set, point):

    def overlaplambda(m1, m2, shuffle):
        if shuffle: m1 = m1[np.random.permutation(m1.shape[0])]
        return np.sum(m1*m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))
    
    dirs = set['arg_input_dir'].shape[0]
    neu = set.iloc[0]['arg_nb_neurons']
    meanfr = np.zeros((neu, dirs))
    i = 0
    for fr in set['meanfr']:
        meanfr[:,i] = np.array(eval(fr))
        i += 1

    num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    resultsp = np.zeros(num_pairs)
    resultspshuff = np.zeros(num_pairs)
    pair_idx = 0
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            # loop axis are the first ones
            resultsp[pair_idx] = overlaplambda(meanfr[:,dir_idx], meanfr[:,dir_idx2], shuffle = False)
            resultspshuff[pair_idx] = overlaplambda(meanfr[:,dir_idx], meanfr[:,dir_idx2], shuffle = True)
            pair_idx += 1
    
    point['overlap'] = tostore(resultsp)
    point['overlapshuff'] = tostore(resultspshuff)


def analyse_spatialcorr(point, all_pfs):

    def spatialcorrlambda(m1, m2, shuffle):
        if shuffle: m1 = m1[:, np.random.permutation(m1.shape[1])]
        in_both_active = (m1>0).any(axis=1)*(m2>0).any(axis=1)
        m1 = m1[in_both_active]
        m2 = m2[in_both_active]
        if m1.shape[0] != 0:
            return np.mean(np.sum(m1*m2, axis=1)/(np.linalg.norm(m1,axis=1)*np.linalg.norm(m2,axis=1)))
        else: return -1

    dirs = all_pfs.shape[2]
    num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    resultsp = np.zeros(num_pairs)
    resultspshuff = np.zeros(num_pairs)
    pair_idx = 0
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            # loop axis are the first ones
            resultsp[pair_idx] = spatialcorrlambda(all_pfs[:,:,dir_idx], all_pfs[:,:,dir_idx2], shuffle = False)
            resultspshuff[pair_idx] = spatialcorrlambda(all_pfs[:,:,dir_idx], all_pfs[:,:,dir_idx2], shuffle = True)
            pair_idx += 1
    
    point['spatialcorr'] = tostore(resultsp)
    point['spatialcorrshuff'] = tostore(resultspshuff)

def analyse_multispatialcorr(point, all_pfs):

    def multispatialcorrlambda(m1, m2): return np.sum(m1*m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))

    in_both_active = (all_pfs[:,:,0]>0).any(axis=1)*(all_pfs[:,:,1]>0).any(axis=1)
    all_pfs = all_pfs[in_both_active, :, 0:2]
    neu = np.sum(in_both_active)

    num_pairs = int(neu*(neu-1)/2) # Gauss formula
    resultsp = np.zeros((num_pairs,2))
    resultspshuff = np.zeros((num_pairs,2))
    pair_idx = 0
    shuff_idx = 0
    for neu_idx in np.arange(neu):
        for neu_idx2 in np.arange(neu_idx):
            # loop axis are the first ones
            resultsp[pair_idx,0] = multispatialcorrlambda(all_pfs[neu_idx,:,0], all_pfs[neu_idx2,:,0])
            resultsp[pair_idx,1] = multispatialcorrlambda(all_pfs[neu_idx,:,1], all_pfs[neu_idx2,:,1])
            pair_idx += 1
        resultspshuff[shuff_idx:pair_idx,:] = resultsp[shuff_idx:pair_idx,:]
        np.random.shuffle(resultspshuff[shuff_idx:pair_idx,1])
        shuff_idx = pair_idx
    
    point['multispatialcorr'] = tostore(resultsp)
    point['multispatialcorrshuff'] = tostore(resultspshuff)

################ RECRUITMENT ANALYSIS #######################

def analyse_recruitment(row, params):
    
    point = row[params]
    point['classes'] = row['classes']
    analyse_perpcs_perclass(point)
    analyse_fr_perclass(row, point)
    
    # TODO Change this to spatial error
    point['decoding_error'] = [row['tracking_error']]
    point['meanfr'] = row['meanfr']
    point['maxfr'] = row['maxfr']

    analyse_pfsize(row, point)
    analyse_pfsize_perclass(point)
    analyse_coveredarea_perclass(row, point)
    
    return point

def analyse_perpcs_perclass(point):

    neu = point['arg_nb_neurons']
    classes = np.array(eval(point['classes']))

    labels = ['permanent', 'tagged', 'recruited', 'silent']
    for c in np.arange(4):
        point['perpcs_' + labels[c]] = tostore(np.array([np.sum(classes == c)/neu]))
    
def analyse_fr_perclass(row, point):
    
    meanfridx = np.array(eval(row['meanfr']))
    maxfridx = np.array(eval(row['maxfr']))
    classes = np.array(eval(row['classes']))
    pcsidx = np.array(eval(row['pcsidx']))
    neu = row['arg_nb_neurons']

    meanfr = np.zeros(neu)
    maxfr = np.zeros(neu)
    meanfr[pcsidx] = meanfridx
    maxfr[pcsidx] = maxfridx
    
    labels = ['permanent', 'tagged', 'recruited', 'silent']
    for c in np.arange(4):
        point['meanfr_' + labels[c]] = tostore(meanfr[classes == c] if np.any(classes == c) else np.array([0]))
        point['maxfr_' + labels[c]]= tostore(maxfr[classes == c] if np.any(classes == c) else np.array([0]))

def analyse_pfsize(row, point):

    with open(row['pfs']) as res_file:
        t0 = time.time()
        pfs = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    point['pfsize'] = tostore(np.sum(pfs > 0, axis=1)*bin_size)

def analyse_pfsize_perclass(point):

    classes = np.array(eval(point['classes']))
    pfsize = np.array(eval(point['pfsize']))
    
    labels = ['permanent', 'tagged', 'recruited', 'silent']
    for c in np.arange(4):
        point['pfsize_' + labels[c]] = tostore(pfsize[classes == c] if np.any(classes == c) else np.array([0]))

def analyse_coveredarea_perclass(row, point):

    with open(row['pfs']) as res_file:
        t0 = time.time()
        pfs = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    point['coveredarea'] = tostore(np.array([np.sum(np.sum(pfs, axis=0) > 0)/b]))
    classes = np.array(eval(point['classes']))

    labels = ['permanent', 'tagged', 'recruited', 'silent']
    for c in np.arange(4):
        point['coveredarea_' + labels[c]] = tostore(np.array([np.sum(np.sum(pfs[classes == c,:], axis=0) > 0)/b]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dir", type=str, default='NormalizedMiliTorusPCS2',
                        help="Directory to read and write files")
    parser.add_argument("--compute", type=str, default='placecells',
                        help = 'Which thing to analyse to make')
    parser.add_argument("--shuffle", action='store_true', default=False,
                        help="Shuffle the data in some way")

    args = parser.parse_args()

compute = args.compute
shuffle = args.shuffle

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = './data/' + args.dir + '/'
if compute in {'ratemaps_pfs', 'classes'}: 
    filename = basepath + timestr + '-' + args.dir + '-database_df.csv'
else:
    filename = basepath + timestr + '-' + args.dir + '-' + compute + '_df.csv'

if compute == 'database':
    df = compute_database(basepath)
else:
    dbase = load_dataframe('database', basepath)
    cols = list(dbase.columns)
    params = [c for c in cols if c.startswith('arg_')]

# On top of the main database
if compute == 'ratemaps_pfs':
    lambdafunc = lambda x, params: analyse_ratemaps_pfs(x)
    df = compute_per_simulation(dbase, params, lambdafunc)
elif compute == 'classes':
    lambdafunc = lambda x, gb: analyse_classes(x)
    df = compute_across(dbase, 'arg_current_amp', params, lambdafunc)  

# Making a new database
elif compute == 'placecells':
    lambdafunc = lambda x, params: analyse_placecells(x, params)
    df = compute_per_simulation(dbase, params, lambdafunc)
elif compute == 'remapping':
    lambdafunc = lambda x, gb: analyse_remapping(x,gb)
    df = compute_across(dbase, 'arg_input_dir', params, lambdafunc)
elif compute == 'recruitment':
    lambdafunc = lambda x, params: analyse_recruitment(x, params)
    df = compute_per_simulation(dbase, params, lambdafunc)


print("Saving results...")
with open(filename, 'wb') as f:
    df.to_csv(f)