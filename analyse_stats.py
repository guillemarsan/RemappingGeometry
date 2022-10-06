from operator import ne
import pathlib, argparse, json, time
import pickle
import numpy as np
import time
from convexsnn.path import get_path


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

def load_data(dim_vect, red_vect, dir_vect, loadid_vect, func):

    num_dims = dim_vect.shape[0]
    num_reds = red_vect.shape[0]
    num_dirs = dir_vect.shape[0]
    num_loadid = loadid_vect.shape[0]

    patt = "*.json"
    path = pathlib.Path(args.read_dir + str(args.dim_pcs))
    results_files = path.rglob(patt)

    results = np.zeros((num_reds, num_dims, num_dirs, num_loadid), dtype=object)

    for f in results_files:
        with open(f) as res_file:
            f = json.load(res_file)

            dim = f['args']['dim_bbox']
            neu = f['args']['nb_neurons']
            red = neu/dim
            dir = int(f['args']['input_dir'][0])
            loadid = f['args']['load_id']

            if (dim in dim_vect) and (red in red_vect) and (dir in dir_vect) and (loadid in loadid_vect): 
                
                dim_idx = np.argwhere(dim_vect == dim)[0,0]
                red_idx = np.argwhere(red_vect == red)[0,0]
                dir_idx = np.argwhere(dir_vect == dir)[0,0]
                loadid_idx = np.argwhere(loadid_vect == loadid)[0,0]

                results[red_idx, dim_idx, dir_idx, loadid_idx] = func(path, f)

    return results

##################### DATA READ ###################################
def read_perpcs(path,f):
    return f['perpcs']

def read_npcs(path,f):
    return f['perpcs']*f['args']['nb_neurons']

def read_activepcs(path,f):
    return f['pcsidx']

def read_activepcs_th(path,f):
    active_list = f['pcsidx']
    Th = read_matrix(path, f, 'Th')
    return (active_list, Th)

def read_maxfr(path,f):
    return f['maxfr']

def read_meanfr(path,f):
    return f['meanfr']

def read_spikes(path,f,times=False):
    if not times:
        s = read_matrix(path, f, 's')
        file = f['s'][:-6]
    else:
        stimes = read_matrix(path, f, 'stimes')
        stimes = stimes.astype(int)
        s = np.zeros((f['args']['nb_neurons'],f['nb_steps']))
        s[stimes[:,0],stimes[:,1]] = 1
        file = f['stimes'][:-11]
    return (str(path), file, s)

def read_spikebins(path,f):
    spikebins = read_matrix(path,f,'spikebins')
    return spikebins

##################### ANALYSE RESULTS ###################################

def analyse_simple(dict, red_vect, results):
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        dict[key] = results[red_idx,:,:,:].reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict

def analyse_nrooms(dict, dim_vect, red_vect, results):
    nbins=np.arange(num_dirs+2)
    for red_idx, red in enumerate(red_vect):
        for dim_idx, dim  in enumerate(dim_vect):
            neu = red*dim
            results_nrooms = np.zeros((neu, num_loadid))
            for dir_idx, dir in enumerate(dir_vect):
                for loadid_idx, loadid in enumerate(loadid_vect):
                    results_nrooms[results[red_idx, dim_idx, dir_idx,loadid_idx], loadid_idx] += 1 

            hist = np.apply_along_axis(lambda a: np.histogram(a, bins=nbins)[0], 0, results_nrooms)
            key = 'd,n = ' + str(dim) + "," + str(neu)
            dict[key] = hist/neu
    return dict

def analyse_diffs(dict, dim_vect, red_vect, results):
    for red_idx, red in enumerate(red_vect):
        for dim_idx, dim  in enumerate(dim_vect):
            neu = red*dim
            npts = int(np.math.factorial(num_dirs)/(2*np.math.factorial(num_dirs-2)))
            points_vect = np.zeros((2,npts))
            j = 0
            for dir_idx1, _ in enumerate(dir_vect):
                for dir_idx2 in np.arange(dir_idx1):
                    # Other measures
                    # px = np.linalg.norm(np.linalg.norm(results_Theta[:,:,b,0] - results_Theta[:,:,c,0],axis=0),axis=0)/d**2
                    # div = np.sum(np.sign(results[:,b,:] + results[:,c,:]),axis=0)
                    # py = np.mean(np.linalg.norm(results[:,b,:] - results[:,c,:],axis=0)/div)

                    Th1 = results[red_idx,dim_idx,dir_idx1,0][1]
                    Th2 = results[red_idx,dim_idx,dir_idx2,0][1]
                    divpx = np.linalg.norm(Th1)*np.linalg.norm(Th2)
                    px = np.sum(Th1 * Th2)/divpx

                    py = 0
                    s1 = np.zeros((neu,1))
                    s2 = np.zeros((neu,1))
                    for loadid_idx, _ in enumerate(loadid_vect):
                        s1[results[red_idx, dim_idx, dir_idx1, loadid_idx][0]] = 1
                        s2[results[red_idx, dim_idx, dir_idx2, loadid_idx][0]] = 1
                        divpy = np.linalg.norm(s1)*np.linalg.norm(s2)
                        py += np.sum(s1.T @ s2,axis=0)/divpy
                    py = py/num_loadid

                    points_vect[0,j] = px
                    points_vect[1,j] = py
                    j += 1

            key = 'd,n = ' + str(dim) + "," + str(neu)
            dict[key] = points_vect
    return dict

def analyse_maxfr(dict, red_vect, results):
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        maxlambda = lambda l: np.max(np.array(l))
        resultsp = np.vectorize(maxlambda)(results[red_idx,:,:,:])
        dict[key] = resultsp.reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict

def analyse_meanfr(dict, red_vect, results):
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        meanlambda = lambda l: np.mean(np.array(l))
        resultsp = np.vectorize(meanlambda)(results[red_idx,:,:,:])
        dict[key] = resultsp.reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict

def analyse_spikes_spikebins(results):
    b = 100
    p, dp, t, dt, time_steps = get_path(dpcs=2, type='uspiral')

    for red_idx, red in enumerate(red_vect):
        for dim_idx, dim  in enumerate(dim_vect):
            for dir_idx, dir in enumerate(dir_vect):
                for loadid_idx, loadid in enumerate(loadid_vect):
                    neu = red*dim

                    res = results[red_idx, dim_idx, dir_idx, loadid_idx]
                    path = res[0]
                    f = res[1]
                    s = res[2]

                    radius = 1
                    row_col = int(np.sqrt(b))
                    step = 2*radius/row_col
                   
                    rows = np.arange(-radius+step/2, radius, step=step)
                    ptr = rows.shape[0]
                    pts = np.arange(ptr**2)
                    ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T
                    
                    spikebins = np.zeros((neu,b))
                    for i in np.arange(neu):
                        sums = np.zeros(pts.shape[0])
                        for j in pts:
                            dist = np.max(np.abs(np.expand_dims(ppts[:,j],-1)-p), axis=0)
                            consider = np.argwhere(dist < (step/2))
                            sums[j] = np.sum(s[i,consider])/(consider.size*dt) if consider.size != 0 else 0
                        spikebins[i,:] = sums

                    np.savetxt("{0}/{1}-spikebins.csv".format(path,f), spikebins, fmt='%.3e')
                    with open('{0}/{1}.json'.format(path,f), 'r+') as file:
                        data = json.load(file)
                        data['spikebins'] = "%s-spikebins.csv" % f
                        file.seek(0)
                        json.dump(data, file, indent=4)
                        file.truncate()

def analyse_maxsize(dict, results):
    b = 100
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        maxlambda = lambda l: np.max(np.sum(np.array(l)>0,axis=1)/b)
        resultsp = np.vectorize(maxlambda)(results[red_idx,:,:,:])
        dict[key] = resultsp.reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict

def analyse_meansize(dict, results):
    b = 100
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        meanlambda = lambda l: np.mean(np.sum(np.array(l)>0,axis=1)/b)
        resultsp = np.vectorize(meanlambda)(results[red_idx,:,:,:])
        dict[key] = resultsp.reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict

def analyse_reparea(dict, results):
    b = 90
    reparealambda = lambda l: np.sum(np.any(np.array(l),axis=0))/b
    for red_idx, red in enumerate(red_vect):
        key = 'redun = ' + str(red)
        resultsp = np.vectorize(reparealambda)(results[red_idx,:,:,:])
        dict[key] = resultsp.reshape(num_dims, num_dirs*num_loadid).astype('float64')

    return dict



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--num_dims", type=int, default=0,
                        help="Number of dimensions")
    parser.add_argument("--num_reds", type=int, default=0,
                        help="Number of redundancies")                   
    parser.add_argument("--num_dirs", type=int, default=6,
                        help="Number of input directions")
    parser.add_argument("--num_loadids", type=int, default=3,
                        help="Number of bbox loadids used")
    parser.add_argument('--dim_vect', nargs='+', type=int, default=[4, 8, 16, 32],
                        help="Dimension of the bbox")
    parser.add_argument('--red_vect', nargs='+', type=int, default=[16],
                        help="Redundancy of the bbox")
    parser.add_argument('--dir_vect', nargs='+', type=int, default=[0],
                        help="Direction of the input vector")
    parser.add_argument('--loadid_vect', nargs='+', type=int, default=[0],
                        help="LoadID of the bbox vector")
    parser.add_argument("--read_dir", type=str, default='./data/NormalizedMiliTorusPCS',
                        help="Directory to read files")
    parser.add_argument("--write_dir", type=str, default='./data/NormalizedMiliTorusPCS2/',
                        help="Directory to dump output")

    args = parser.parse_args()

compute = 'reparea'

timestr = time.strftime("%Y%m%d-%H%M%S")
name = "pcs-" + str(args.dim_pcs)
basepath = args.write_dir + timestr + "-" + name


dim_vect = 2**(np.arange(args.num_dims)+2) if args.num_dims != 0 else np.array(args.dim_vect)
num_dims = dim_vect.shape[0]
red_vect = 2**(np.arange(args.num_reds)+2) if args.num_reds != 0 else np.array(args.red_vect)
num_reds = red_vect.shape[0]
dir_vect = np.arange(args.num_dirs) if args.num_dirs != 0 else np.array(args.dir_vect)
num_dirs = dir_vect.shape[0]
loadid_vect = np.arange(args.num_loadids) if args.num_loadids != 0 else np.array(args.loadid_vect)
num_loadid = loadid_vect.shape[0]

if compute == 'perpcs':
    load_func = lambda path, f: read_perpcs(path, f)
elif compute == 'npcs':
    load_func = lambda path, f: read_npcs(path,f)
elif compute == 'nrooms':
    load_func = lambda path, f: read_activepcs(path, f)
elif compute == 'diffs':
    load_func = lambda path, f: read_activepcs_th(path, f)
elif compute == 'maxfr':
    load_func = lambda path, f: read_maxfr(path,f)
elif compute == 'meanfr':
    load_func = lambda path, f: read_meanfr(path,f)
elif compute == 'spikebins':
    load_func = lambda path, f: read_spikes(path,f,times=True)
elif compute in {'maxsize', 'meansize', 'reparea'}:
    load_func = lambda path, f: read_spikebins(path,f)

print("Loading data...")
results = load_data(dim_vect, red_vect, dir_vect, loadid_vect, func=load_func)

patt = "*-%s_dict.pkl" % compute
path = pathlib.Path(args.read_dir + str(args.dim_pcs))
results_files = path.rglob(patt)
no_load = False
try:
    file = next(results_files)
except StopIteration:
    no_load = True
if no_load:
    dict = {}
else:
    file_name = str(file)
    with open(file_name, 'rb') as f:
        dict = pickle.load(f)

print ("Analysing data...")
dict_save = False
if compute in {'perpcs', 'npcs'}:
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_simple(dict, red_vect, results)
    dict_save = True

elif compute == 'nrooms':
    if no_load: dict['xaxis'] = np.arange(num_dirs+2)[:-1]
    dict = analyse_nrooms(dict, dim_vect, red_vect, results)
    dict_save = True

elif compute == 'diffs':
    dict = analyse_diffs(dict, dim_vect, red_vect, results)
    dict_save = True

elif compute == 'maxfr':
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_maxfr(dict, red_vect, results)
    dict_save = True

elif compute == 'meanfr':
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_meanfr(dict, red_vect, results)
    dict_save = True

elif compute == 'spikebins':
    analyse_spikes_spikebins(results)

elif compute == 'maxsize':
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_maxsize(dict, results)
    dict_save = True

elif compute == 'meansize':
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_meansize(dict, results)
    dict_save = True

elif compute == 'reparea':
    if no_load: dict['xaxis'] = dim_vect
    dict = analyse_reparea(dict, results)
    dict_save = True

if dict_save:
    print("Saving results...")
    with open('{0}-{1}_dict.pkl'.format(basepath,compute), 'wb') as f:
        pickle.dump(dict, f)
 