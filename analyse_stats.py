from operator import ne
import pathlib, argparse, json, time
from utils import compute_meshgrid, compute_ratemap_s, compute_ratemap_r, compute_pf, compute_pathloc
import numpy as np
import time
import pandas as pd
from convexsnn.path import get_path
from convexsnn.network import get_model
from convexsnn.AngleEncoder import AngleEncoder

##################### STANDARDS ###################################

b = 400 #number of bins
radius = 1 #radius of the environment
area = (2*radius)**2
bin_size = area/b #bin_size
tostore = lambda array: np.array2string(array, separator=',', suppress_small=True, threshold=np.inf, max_line_width=np.inf)

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
    results_files = sorted(path.rglob(patt), reverse=True)
    no_load = False
    if len(results_files) > 0:
        file = results_files[0]
    else:
        no_load = True

    if no_load:
        df = pd.DataFrame()
    else:
        old_file_name = str(file)
        with open(old_file_name, 'rb') as f:
            df = pd.read_csv(f, index_col=[0])

    return df

def read_pfs(set, neu, b, dirs, ratemaps=False):

    all_pfs = np.zeros((neu, b, dirs))
    i = 0
    filelist = set['pfs'] if not ratemaps else set['ratemaps']
    for file in filelist:
        with open(file) as res_file:
            t0 = time.time()
            pfs = np.genfromtxt(res_file)
            t1 = time.time()
            print(t1-t0)
        all_pfs[:,:,i] = pfs
        i += 1

    return all_pfs

def read_trajs(set):

    i = 0
    all = [[],[],[],[]]
    for type in ['x', 'xhat', 'g', 'ghat']:
        filelist = set[type]
        for file in filelist:
            with open(file) as res_file:
                a = np.genfromtxt(res_file)
            all[i].append(a)
        all[i] = np.stack(all[i], axis=2)
        i += 1

    return all


def read_D(point, neu, dbbox):

    D = np.zeros((neu, dbbox))
    sepnorm = None if len(point['arg_model_conj']) == 1 else point['arg_dim_pcs']*2
    _, D, _ = get_model(dbbox ,neu, dbbox, connectivity=point['arg_model'], decod_amp=point['arg_decoder_amp'], 
                    load_id=point['arg_load_id'], conn_seed=point['arg_conn_seed'], lognor_seed=point['arg_lognor_seed'], 
                    lognor_sigma=point['arg_lognor_sigma'], rnn=point['arg_rnn'], conj=point['arg_model_conj'],
                    sepnorm=sepnorm)
    
    return D

def read_inputs(set):

    xs = []
    i = 0
    filelist = set['x']
    for file in filelist:
        with open(file) as res_file:
            x = np.genfromtxt(res_file)
        xs.append(x) 
        i += 1
    xs = np.stack(xs,axis=2)
    return xs

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

            delete_list = ['arg_compute_fr', 'arg_dir', 'arg_plot', 'arg_gif', 'arg_save']
            for d in delete_list:
                del f[d]

            newdata.append(f)

    return pd.DataFrame(newdata)

def compute_per_simulation(dbase, params, func):
    return dbase.apply(lambda x: func(x, params), axis=1)

def compute_across(dbase, across, params, func):

    groupby_pars = params
    for a in across:
        groupby_pars.remove(a)

    gb = dbase.groupby(groupby_pars, as_index=False)
    df = gb.apply(lambda x: func(x, groupby_pars))
    return df


##################### ANALYSE RESULTS ###################################

########## ADD TO MAIN DATABASE #############

def analyse_ratemaps_pfs(point):
    spikes = point['arg_simulate'] != 'minimization'
    if spikes: return analyse_ratemaps_pfs_s(point)
    else: return analyse_pfs_r(point)

def analyse_pfs_r(point):
    id = point['basepath']
    rates_name = "{0}-rates.csv".format(id)
    with open(rates_name) as res_file:
        t0 = time.time()
        rates = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    path_tmax = point['arg_path_tmax']
    path_type = point['arg_path_type']
    path_seed = point['arg_path_seed']

    dpcs = point['arg_dim_pcs']
    p, dp, t, dt, time_steps = get_path(dpcs=dpcs, type=path_type, tmax=path_tmax, path_seed=path_seed)
    bins = compute_meshgrid(radius,b,dpcs)
    neu = point['arg_nb_neurons']

    # Compute occupancies    
    pathloc = compute_pathloc(p, bins, dpcs)
    tb = np.sum(pathloc, axis=1)*dt
    occupancies = tb/np.max(t)

    ratemaps = np.zeros((neu,b))
    placefields = np.zeros((neu,b))
    meanfr = np.zeros(neu)
    pfsize = np.zeros(neu)
    # spikes = np.zeros(neu)
    for i in np.arange(neu):
        ratemaps[i,:] = compute_ratemap_r(rates[i,:], pathloc)
        placefields[i,:] = ratemaps[i,:].copy()
        placefields[i, placefields[i,:] < 1e-3] = 0 # for analysis
        meanfr[i] = np.mean(placefields[i,:])
        pfsize[i] = bin_size*np.sum(placefields[i,:] > 0)
        # spikes[i] = np.sum(s[i,:])

    pcsidx = np.where(meanfr > 0)[0]
    perpcs = pcsidx.shape[0]/neu
    meanperpfsize = np.mean(pfsize[pfsize > 0])/area

    # point['spikes'] = tostore(spikes)
    point['meanfr'] = tostore(meanfr)
    point['pfsize'] = tostore(pfsize)
    point['meanperpfsize'] = meanperpfsize
    point['occupancies'] = tostore(occupancies)
    point['pcsidx'] = tostore(pcsidx)
    point['perpcs'] = perpcs

    ratemaps_name = "{0}-ratemaps.csv".format(id)
    pfs_name = "{0}-pfs.csv".format(id)
    np.savetxt(ratemaps_name, ratemaps, fmt='%.3e')
    np.savetxt(pfs_name, placefields, fmt='%.3e')
    point['ratemaps'] = ratemaps_name
    point['pfs'] = pfs_name
    if point['arg_save_input']: 
        point['x'] = "{0}-x.csv".format(id)
        point['xhat'] = "{0}-xhat.csv".format(id)
        point['g'] = "{0}-g.csv".format(id)
        point['ghat'] = "{0}-ghat.csv".format(id)
    

    return point

def analyse_ratemaps_pfs_s(point):

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

    path_tmax = point['arg_path_tmax']
    path_type = point['arg_path_type']
    path_seed = point['arg_path_seed']

    dpcs = point['arg_dpcs']
    p, dp, t, dt, time_steps = get_path(dpcs=dpcs, type=path_type, tmax=path_tmax, path_seed=path_seed)
    bins = compute_meshgrid(radius,b,dpcs)
    neu = point['arg_nb_neurons']

    # Compute occupancies    
    pathloc = compute_pathloc(p, bins, dpcs)
    tb = np.sum(pathloc, axis=1)*dt
    occupancies = tb/np.max(t)

    ratemaps = np.zeros((neu,b))
    placefields = np.zeros((neu,b))
    meanfr = np.zeros(neu)
    pfsize = np.zeros(neu)
    spikes = np.zeros(neu)
    for i in np.arange(neu):
        ratemaps[i,:] = compute_ratemap_s(s[i,:], pathloc, tb)
        placefields[i,:] = compute_pf(ratemaps[i,:], bins)
        meanfr[i] = np.sum(occupancies*placefields[i,:])
        pfsize[i] = bin_size*np.sum(placefields[i,:] > 0)
        spikes[i] = np.sum(s[i,:])

    pcsidx = np.where(meanfr > 0)[0]
    perpcs = pcsidx.shape[0]/neu
    meanperpfsize = np.mean(pfsize[pfsize > 0])/area

    point['spikes'] = tostore(spikes)
    point['meanfr'] = tostore(meanfr)
    point['pfsize'] = tostore(pfsize)
    point['meanperpfsize'] = meanperpfsize
    point['occupancies'] = tostore(occupancies)
    point['pcsidx'] = tostore(pcsidx)
    point['perpcs'] = perpcs

    ratemaps_name = "{0}-ratemaps.csv".format(id)
    pfs_name = "{0}-pfs.csv".format(id)
    np.savetxt(ratemaps_name, ratemaps, fmt='%.3e')
    np.savetxt(pfs_name, placefields, fmt='%.3e')
    point['ratemaps'] = ratemaps_name
    point['pfs'] = pfs_name
    if point['arg_save_input']: point['x'] = "{0}-x.csv".format(id)

    return point

def analyse_classes(set):

    neu = set.iloc[0]['arg_nb_neurons']
    
    # [TagPermanent, TagMuted, | nTagPermanent, nTagMuted, | SilentPermanent, SilentRecruited]
    classes = np.zeros(neu) 
    original = set[(set['arg_tagging_sparse']==0)].iloc[0]

    active = np.array(eval(original['pcsidx']))
    silent = np.setdiff1d(np.arange(neu),active)

    classes[active] = 0
    classes[silent] = 4

    set.at[0,'classes'] = tostore(classes)

    for i in np.arange(1,len(set)):
        
        casenow = set.iloc[i]
        tagged = np.array(eval(casenow['arg_tagged_idx']),dtype=int)
        ntagged = np.setdiff1d(active, tagged)
        activenow = np.array(eval(casenow['pcsidx']))
        silentnow = np.setdiff1d(np.arange(neu),activenow)

        tagPerm = np.intersect1d(activenow, tagged)
        tagMuted = np.intersect1d(silentnow, tagged)
        ntagPerm = np.intersect1d(activenow, ntagged)
        ntagMuted = np.intersect1d(silentnow, ntagged)
        sPerm = np.intersect1d(silentnow, silent)
        sRecruited = np.intersect1d(activenow, silent)

        classes[tagPerm] = 0
        classes[tagMuted] = 1
        classes[ntagPerm] = 2
        classes[ntagMuted] = 3
        classes[sPerm] = 4
        classes[sRecruited] = 5

        set.at[i,'classes'] = tostore(classes)

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
    info_bins = np.linspace(0+1e-5,np.max(results_spatialinfo),10)
    info_bins = np.insert(info_bins,0,0)
    
    active_neurons = np.array(eval(row['pcsidx']))
    hist = np.histogram(results_spatialinfo[active_neurons], bins=info_bins)[0]

    point['spatialinfo'] = tostore(hist)
    point['spatialinfo_bins'] = tostore(np.array(info_bins))

    return point



####### REMAPPING ANALYSIS ###########

def analyse_remapping(set, params):

    point = set.iloc[0][params]
    neu = set.iloc[0]['arg_nb_neurons']
    dirs = len(set)

    dbbox = set.iloc[0]['arg_dim_bbox']
    D = read_D(point, neu, dbbox)
    all_ratemaps = read_pfs(set, neu, b, dirs, ratemaps=True)

    analyse_combine_stats(set, point)

    if point['arg_dim_pcs'] == 1:
        analyse_pca(point, all_ratemaps)
        all_trajs = read_trajs(set)
        analyse_store_trajs(point, all_trajs)
        analyse_store_D(point, D)
    
    analyse_remapping_vector(point, all_ratemaps, D)
    analyse_nrooms(set, point)
    analyse_overlap(set, point)
    
    all_pfs = read_pfs(set, neu, b, dirs)

    analyse_spatialcorr(point, all_pfs)
    analyse_multispatialcorr(point, all_pfs)

    analyse_frdistance_pertuning(set, point, D)
    analyse_spatialcorr_pertuning(point, all_pfs, D)
    analyse_nrooms_pertuning(point, all_pfs, D)
    return point

def alignment(a,b):
    if a.ndim == 1: return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))
    else: return np.sum(a*b, axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))

def distance(a,b):
    return np.abs(a - b)

def analyse_combine_stats(set,point):

    dirs = set['arg_env'].shape[0]
    dbbox = set.iloc[0]['arg_dim_bbox']
    dpcs = set.iloc[0]['arg_dim_pcs']
    denvs = (dbbox/2) - dpcs
    meanperpfsizes = np.zeros(dirs)
    perpcs = np.zeros(dirs)
    dimp_errors = np.zeros(dirs)
    dime_errors = np.zeros(dirs)
    dimg_errors = np.zeros(dirs)
    for i in np.arange(dirs):
        meanperpfsizes[i] = set.iloc[i]['meanperpfsize']
        perpcs[i] = set.iloc[i]['perpcs']
        dimg_errors[i] = set.iloc[i]['g_error']/np.sqrt(dpcs)
        if set.iloc[0]['arg_encoding'] in {'flexible', 'parallel', 'flexibleGP', 'flexibler'}: 
            dimp_errors[i] = set.iloc[i]['p_error']/np.sqrt(dpcs)
            dime_errors[i] = set.iloc[i]['e_error']/np.sqrt(denvs)

    point['meanperpfsizes'] = tostore(meanperpfsizes)
    point['perpcs'] = tostore(perpcs)
    point['dimg_errors'] = tostore(dimg_errors)
    point['dimp_errors'] = tostore(dimp_errors)
    point['dime_errors'] = tostore(dime_errors)

def analyse_pca(point, all_ratemaps):

    def pca(x):
        centered = x - np.mean(x,axis=0)
        cov = np.cov(centered, rowvar=False)
        lmb, v = np.linalg.eigh(cov)
        idx = np.argsort(lmb)[::-1]
        v = v[:, idx]
        varexp = lmb[idx]/np.sum(lmb)
        return varexp, v

    envs = all_ratemaps.shape[2]
    time_steps = all_ratemaps.shape[1]

    flatten_r = np.hstack([all_ratemaps[:,:,i] for i in np.arange(envs)]).T

    varexp_r, v_r = pca(flatten_r) 
    aux = np.dot(flatten_r, v_r[:,:3])

    pca_proj_r = np.zeros((3,time_steps,envs))
    for i in np.arange(envs):
        pca_proj_r[:,:,i] = aux[i*time_steps:(i+1)*time_steps,:].T
    
    point['var_explained_r'] = tostore(varexp_r)
    point['pca_proj_r'] = tostore(pca_proj_r)

def analyse_store_D(point, D):
    point['D'] = tostore(D)

def analyse_store_trajs(point, all_trajs):

    point['x_trajs'] = tostore(all_trajs[0])
    point['xhat_trajs'] = tostore(all_trajs[1])
    point['g_trajs'] = tostore(all_trajs[2])
    point['ghat_trajs'] = tostore(all_trajs[3])

def analyse_remapping_vector(point, all_ratemaps, D):

    dirs = all_ratemaps.shape[2]
    ps = point['arg_encoding'] not in {'sensory'}
    es = False
    if point['arg_encoding'] in {'flexible', 'flexibleGP', 'flexibler', 'parallel'}: 
        es = True
        edims = point['arg_dim_pcs']*2
    elif point['arg_encoding'] in {'sensory', 'sensoryGP'}:
        es = True
        edims = 0
    DR = np.linalg.pinv(D)

    num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    norm_remap = np.zeros(num_pairs)
    norm_Dy = np.zeros(num_pairs)
    norm_Dye = np.zeros(num_pairs)
    norm_nu = np.zeros(num_pairs)
    pair_idx = 0
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            # loop axis are the first ones
            remap_vec = all_ratemaps[:,:,dir_idx] - all_ratemaps[:,:,dir_idx2]
            norm_remap[pair_idx] = np.linalg.norm(remap_vec)
            dy = D @ remap_vec
            if ps: norm_Dy[pair_idx] = np.linalg.norm(DR @ dy) if not es else np.linalg.norm(DR[:,:edims] @ dy[:edims,:])
            if es: norm_Dye[pair_idx] = np.linalg.norm(DR[:,edims:] @ dy[edims:,:])
            dnu = remap_vec - DR @ dy
            norm_nu[pair_idx] = np.linalg.norm(dnu)
            pair_idx += 1

    point['norm_remap'] = tostore(norm_remap)
    point['norm_Dy'] = tostore(norm_Dy)
    point['norm_Dye'] = tostore(norm_Dye)
    point['norm_nu'] = tostore(norm_nu)

def analyse_nrooms(set, point):

    neus = set.iloc[0]['arg_nb_neurons']
    dirs = set['arg_env'].shape[0]

    nbins=np.arange(dirs+2)
    results_nrooms = np.zeros(int(neus))
    for pcsidx in set['pcsidx']:
        results_nrooms[np.array(eval(pcsidx))] += 1 

    hist = np.histogram(results_nrooms, bins=nbins)[0].astype(float)
    
    point['nrooms'] = tostore(hist)
    point['nrooms_bins'] = tostore(np.arange(dirs+1)) 

def analyse_overlap(set, point):

    def overlapshufflambda(m1,m2):
        resshuff = 0
        for _ in np.arange(20):
            shuff = m1[np.random.permutation(m1.shape[0])]
            aux = alignment(shuff, m2)
            if aux is not None:
                resshuff += aux
        return resshuff/20

    # Read meanfr
    dirs = set['arg_env'].shape[0]
    neu = set.iloc[0]['arg_nb_neurons']
    meanfr = np.zeros((neu, dirs))
    i = 0
    for fr in set['meanfr']:
        meanfr[:,i] = np.array(eval(fr))
        i += 1

    # Only neurons that are active in at least one environment (discard silent)
    atleast_one = np.any(meanfr,axis=1)
    meanfr = meanfr[atleast_one,:]

    num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    resultsp = np.zeros(num_pairs)
    resultspbin = np.zeros(num_pairs)
    resultspshuff = np.zeros(num_pairs)
    resultspbinshuff = np.zeros(num_pairs)
    pair_idx = 0
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            # loop axis are the first ones
            resultsp[pair_idx] = alignment(meanfr[:,dir_idx], meanfr[:,dir_idx2])
            resultspbin[pair_idx] = alignment(meanfr[:,dir_idx] > 0, meanfr[:,dir_idx2] >0)
            resultspshuff[pair_idx] = overlapshufflambda(meanfr[:,dir_idx], meanfr[:,dir_idx2])
            resultspbinshuff[pair_idx] = overlapshufflambda(meanfr[:,dir_idx] > 0, meanfr[:,dir_idx2] >0)
            pair_idx += 1
    
    point['overlap'] = tostore(resultsp)
    point['overlapbin'] = tostore(resultspbin)
    point['overlapshuff'] = tostore(resultspshuff)
    point['overlapbinshuff'] = tostore(resultspbinshuff)


def analyse_spatialcorr(point, all_pfs):

    def spatialcorrlambda(m1, m2):
        in_both_active = (m1>0).any(axis=1)*(m2>0).any(axis=1)
        if np.any(in_both_active):
            m1 = m1[in_both_active]
            m2 = m2[in_both_active]
            return np.mean(alignment(m1,m2))
        else: return None

    def spatialcorrshufflambda(m1,m2):
        resshuff = 0
        i = 0
        for _ in np.arange(20):
            shuff = m1[:, np.random.permutation(m1.shape[1])]
            aux = spatialcorrlambda(shuff, m2)
            if aux is not None:
                resshuff += aux
                i += 1
        return resshuff/i if i != 0 else None
    
    # Only environments that have a common active neuron
    dirs = all_pfs.shape[2]
    # num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    # aneuron_incommon = np.zeros(num_pairs)
    # i = 0
    # for dir_idx in np.arange(dirs):
    #     for dir_idx2 in np.arange(dir_idx):
    #         aneuron_incommon[i] = np.any((all_pfs[:,:,dir_idx] > 0).any(axis=1)*(all_pfs[:,:,dir_idx2] > 0).any(axis=1))
    #         i += 1

    # anum_pairs = np.sum(aneuron_incommon)        
    # resultsp = np.zeros(anum_pairs)
    # resultspshuff = np.zeros(anum_pairs)
    # pair_idx = 0
    # i = 0

    resultsp = np.array([])
    resultspshuff = np.array([])
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            res = spatialcorrlambda(all_pfs[:,:,dir_idx], all_pfs[:,:,dir_idx2])
            if res is not None:
                resultsp = np.append(resultsp, res) 

            resshuff = spatialcorrshufflambda(all_pfs[:,:,dir_idx], all_pfs[:,:,dir_idx2])
            if resshuff is not None:
                resultspshuff = np.append(resultspshuff, resshuff)     
    
    point['spatialcorr'] = tostore(resultsp)
    point['spatialcorrshuff'] = tostore(resultspshuff)

def analyse_multispatialcorr(point, all_pfs):

    # def multispatialcorrlambda(m1, m2): return np.sum(m1*m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))
    
    # TODO Check that the first pair of environments have at least two active place cells in common

    # Only neurons that are both active in env 0 and 1
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
            resultsp[pair_idx,0] = alignment(all_pfs[neu_idx,:,0], all_pfs[neu_idx2,:,0])
            resultsp[pair_idx,1] = alignment(all_pfs[neu_idx,:,1], all_pfs[neu_idx2,:,1])
            pair_idx += 1
        resultspshuff[shuff_idx:pair_idx,:] = resultsp[shuff_idx:pair_idx,:]
        np.random.shuffle(resultspshuff[shuff_idx:pair_idx,1])
        shuff_idx = pair_idx
    
    point['multispatialcorr'] = tostore(resultsp)
    point['multispatialcorrshuff'] = tostore(resultspshuff)

def analyse_frdistance_pertuning(set, point, D):

      
    dirs = set['arg_env'].shape[0]
    neu = set.iloc[0]['arg_nb_neurons']
    meanfr = np.zeros((neu, dirs))
    i = 0
    for fr in set['meanfr']:
        meanfr[:,i] = np.array(eval(fr))
        i += 1

    # Only neurons that are active in at least one environment (discard silent)
    atleast_one = np.any(meanfr,axis=1)
    meanfr = meanfr[atleast_one,:]

    atleast_one_idx = np.argwhere(atleast_one)
    num_atleast_one = np.sum(atleast_one)

    resultsp = np.zeros((num_atleast_one,2))
    resultspshuff = np.zeros((num_atleast_one,2))
    num_pairs = int(dirs*(dirs-1)/2)
    for neu_idx in np.arange(num_atleast_one):
        distances = np.zeros((num_pairs))
        i = 0
        for dir_idx in np.arange(dirs):
            for dir_idx2 in np.arange(dir_idx):
                distances[i] = distance(meanfr[neu_idx,dir_idx], meanfr[neu_idx,dir_idx2])
                i += 1
        resultsp[neu_idx,1] = np.mean(distances)
        resultsp[neu_idx,0] = np.linalg.norm(D[:4, atleast_one_idx[neu_idx]])
    resultspshuff = resultsp.copy()
    np.random.shuffle(resultspshuff[:,0])
    
    point['frdistance_pertuning'] = tostore(resultsp)
    point['frdistance_pertuningshuff'] = tostore(resultspshuff)

def analyse_spatialcorr_pertuning(point, all_pfs, D):

    def spatialcorr_pertuninglambda(m1):
        dirs = m1.shape[2]
        overlaps = np.array([])
        for dir_idx in np.arange(dirs):
            for dir_idx2 in np.arange(dir_idx):
                in_both_active = np.any(m1[0,:,dir_idx])*np.any(m1[0,:,dir_idx2])
                if in_both_active:
                    overlaps = np.append(overlaps, alignment(m1[0,:,dir_idx], m1[0,:,dir_idx2])) 
        return np.mean(overlaps)
    
    
    dirs = all_pfs.shape[2]
    neu = all_pfs.shape[0]
    num_pairs = int(dirs*(dirs-1)/2) # Gauss formula
    neurons_incommon = np.zeros((num_pairs, neu))
    i = 0
    for dir_idx in np.arange(dirs):
        for dir_idx2 in np.arange(dir_idx):
            neurons_incommon[i,:] = (all_pfs[:,:,dir_idx] > 0).any(axis=1)*(all_pfs[:,:,dir_idx2] > 0).any(axis=1)
            i += 1

    # Only neurons that are active in at least one pair of environments
    neurons_atleast = np.argwhere(np.any(neurons_incommon, axis=0))
    num_neurons_atleast = neurons_atleast.shape[0]

    resultsp = np.zeros((num_neurons_atleast,2))
    resultspshuff = np.zeros((num_neurons_atleast,2))
    for neu_idx in np.arange(num_neurons_atleast):
        resultsp[neu_idx,1] = spatialcorr_pertuninglambda(all_pfs[neurons_atleast[neu_idx],:,:])
        resultsp[neu_idx,0] = np.linalg.norm(D[:4, neurons_atleast[neu_idx]])
    resultspshuff = resultsp.copy()
    np.random.shuffle(resultspshuff[:,0])
    
    point['spatialcorr_pertuning'] = tostore(resultsp)
    point['spatialcorr_pertuningshuff'] = tostore(resultspshuff)

def analyse_nrooms_pertuning(point, all_pfs, D):

    neu = all_pfs.shape[0]
    
    resultsp = np.zeros((neu,2))
    resultspshuff = np.zeros((neu,2))
    for neu_idx in np.arange(neu):
        resultsp[neu_idx,1] = np.sum(np.any(all_pfs[neu_idx,:,:], axis=0))
        resultsp[neu_idx,0] = np.linalg.norm(D[:4,neu_idx])
    resultspshuff = resultsp.copy()
    np.random.shuffle(resultspshuff[:,0])
    
    point['nrooms_pertuning'] = tostore(resultsp)
    point['nrooms_pertuningshuff'] = tostore(resultspshuff)


################ RECRUITMENT ANALYSIS #######################

def analyse_recruitment(row, params):
    
    point = row[params]
    point['classes'] = row['classes']
    analyse_perpcs_perclass(point)
    
    # TODO Change this to spatial error?
    point['g_error'] = [row['g_error']]

    analyse_pcsmfr(row, point)
    analyse_mfr_perclass(row, point)

    analyse_pcspfsize(row, point)
    analyse_pfsize_perclass(row, point)

    point['spikes'] = [np.sum(np.array(eval(row['spikes'])))]
    analyse_spikes_perclass(row, point)

    analyse_coveredarea_perclass(row, point)
    
    return point

def analyse_perpcs_perclass(point):

    neu = point['arg_nb_neurons']
    classes = np.array(eval(point['classes']))

    labels = ['tP', 'tM', 'ntP', 'ntM', 'sP', 'sR']
    for c in np.arange(6):
        point['perpcs_' + labels[c]] = tostore(np.array([np.sum(classes == c)/neu]))
    
def analyse_mfr_perclass(row, point):
    
    meanfr = np.array(eval(row['meanfr']))
    classes = np.array(eval(row['classes']))
    
    labels = ['tP', 'tM', 'ntP', 'ntM', 'sP', 'sR']
    for c in np.arange(6):
        point['meanfr_' + labels[c]] = tostore(meanfr[classes == c] if np.any(classes == c) else np.array([0]))

def analyse_pcsmfr(row, point):

    pcsidx = np.array(eval(row['pcsidx']))
    meanfr = np.array(eval(row['meanfr']))

    point['pcsmeanfr'] = tostore(meanfr[pcsidx])

def analyse_pcspfsize(row, point):

    pcsidx = np.array(eval(row['pcsidx']))
    pfsize = np.array(eval(row['pfsize']))
    
    point['pcspfsize'] = tostore(pfsize[pcsidx])


def analyse_pfsize_perclass(row, point):

    classes = np.array(eval(row['classes']))
    pfsize = np.array(eval(row['pfsize']))
    
    labels = ['tP', 'tM', 'ntP', 'ntM', 'sP', 'sR']
    for c in np.arange(6):
        point['pfsize_' + labels[c]] = tostore(pfsize[classes == c] if np.any(classes == c) else np.array([0]))

def analyse_spikes_perclass(row, point):

    classes = np.array(eval(row['classes']))
    spikes = np.array(eval(row['spikes']))
    
    labels = ['tP', 'tM', 'ntP', 'ntM', 'sP', 'sR']
    for c in np.arange(6):
        point['spikes_' + labels[c]] = [np.sum(spikes[classes == c])] if np.any(classes == c) else [0]

def analyse_coveredarea_perclass(row, point):

    with open(row['pfs']) as res_file:
        t0 = time.time()
        pfs = np.genfromtxt(res_file)
        t1 = time.time()
        print(t1-t0)

    point['coveredarea'] = [np.sum(np.sum(pfs, axis=0) > 0)/b]

    classes = np.array(eval(row['classes']))

    labels = ['tP', 'tM', 'ntP', 'ntM', 'sP', 'sR']
    for c in np.arange(6):
        point['coveredarea_' + labels[c]] = [np.sum(np.sum(pfs[classes == c,:], axis=0) > 0)/b]

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dir", type=str, default='gridcellsd1',
                        help="Directory to read and write files")
    parser.add_argument("--dir_loc", type=str, default='./data/v1',
                        help="Location of the directory" )
    parser.add_argument("--compute", type=str, default='remapping',
                        help = 'Which thing to analyse to make')
    parser.add_argument("--null", action='store_true', default=False,
                        help = 'If nullspace remapping or not')
    parser.add_argument("--shuffle", action='store_true', default=False,
                        help="Shuffle the data in some way")

    args = parser.parse_args()

compute = args.compute
shuffle = args.shuffle

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = args.dir_loc + '/' + args.dir + '/'
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
    df = compute_across(dbase, ['arg_tagging_sparse','arg_current_amp','arg_tagged_idx'], params, lambdafunc)  

# Making a new database
elif compute == 'placecells':
    lambdafunc = lambda x, params: analyse_placecells(x, params)
    df = compute_per_simulation(dbase, params, lambdafunc)
elif compute == 'remapping':
    lambdafunc = lambda x, gb: analyse_remapping(x,gb)
    df = compute_across(dbase, ['arg_env'] if not args.null else 
                        ['arg_tagging_sparse','arg_current_amp','arg_tagged_idx'], params, lambdafunc)
elif compute == 'recruitment':
    lambdafunc = lambda x, params: analyse_recruitment(x, params)
    df = compute_per_simulation(dbase, params, lambdafunc)


print("Saving results...")
with open(filename, 'wb') as f:
    df.to_csv(f)