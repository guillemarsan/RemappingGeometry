from operator import ne
import pathlib, argparse, json, time
import pickle
import numpy as np
import time

from convexsnn.plot import plot_errorplot, plot_scatterplot


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
    parser.add_argument('--red_vect', nargs='+', type=int, default=[16,32,64],
                        help="Redundancy of the bbox")
    parser.add_argument('--dir_vect', nargs='+', type=int, default=[0],
                        help="Direction of the input vector")
    parser.add_argument('--loadid_vect', nargs='+', type=int, default=[0],
                        help="LoadID of the bbox vector")
    parser.add_argument("--read_dir", type=str, default='./data/DBTorusPCS',
                        help="Directory to read files")
    parser.add_argument("--write_dir", type=str, default='./out/',
                        help="Directory to dump output")

    args = parser.parse_args()

plot = 'meansize'

dim_vect = 2**(np.arange(args.num_dims)+2) if args.num_dims != 0 else np.array(args.dim_vect)
num_dims = dim_vect.shape[0]
red_vect = 2**(np.arange(args.num_reds)+2) if args.num_reds != 0 else np.array(args.red_vect)
num_reds = red_vect.shape[0]
dir_vect = np.arange(args.num_dirs) if args.num_dirs != 0 else np.array(args.dir_vect)
num_dirs = dir_vect.shape[0]
loadid_vect = np.arange(args.num_loadids) if args.num_loadids != 0 else np.array(args.loadid_vect)
num_loadid = loadid_vect.shape[0]

timestr = time.strftime("%Y%m%d-%H%M%S")
name = "pcs-" + str(args.dim_pcs)
basepath = args.write_dir + timestr + "-" + name

print("Loading results...")

patt = "*-%s_dict.pkl" % plot
path = pathlib.Path(args.read_dir + str(args.dim_pcs))
results_files = path.rglob(patt)
file = next(results_files)
file_name = str(file)
with open(file_name, 'rb') as f:
    dict = pickle.load(f)

print ("Plotting results...")
if plot == 'perpcs':

    allowed_keys = []
    for red in red_vect:
        allowed_keys.append("redun = " + str(red))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value

    title = 'Percentage of place cells for PCs in ' + str(args.dim_pcs) + 'D'
    xaxis = dict['xaxis']
    labels = ['Dimension', 'Percentage of place cells'] 
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath)

elif plot == 'nrooms':

    allowed_keys = []
    for red in red_vect:
        for dim in dim_vect:
            neu = red*dim
            allowed_keys.append('d,n = ' + str(dim) + "," + str(neu))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value
    
    title = 'Percentage of neurons active in n rooms'
    xaxis = dict['xaxis']
    labels = ['Number of rooms', 'Percentage of PCs']
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath)

elif plot == 'diffs':

    allowed_keys = []
    for red in red_vect:
        for dim in dim_vect:
            neu = red*dim
            allowed_keys.append('d,n = ' + str(dim) + "," + str(neu))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value
    
    title = 'Relationship between angle in embeddings and active cells'
    labels = ['Angle in embedding', 'Angle in active cells']
    print("Plotting results...")
    plot_scatterplot(dict2, title, labels, basepath)

elif plot == 'maxfr':

    allowed_keys = []
    for red in red_vect:
        allowed_keys.append("redun = " + str(red))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value

    title = 'Maximum firing rate'
    xaxis = dict['xaxis']
    labels = ['Dimension', 'Max FR'] 
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath, ynormalized=False)

elif plot == 'meanfr':

    allowed_keys = []
    for red in red_vect:
        allowed_keys.append("redun = " + str(red))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value

    title = 'Mean firing rate'
    xaxis = dict['xaxis']
    labels = ['Dimension', 'Mean FR'] 
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath, ynormalized=False)

elif plot == 'maxsize':

    allowed_keys = []
    for red in red_vect:
        allowed_keys.append("redun = " + str(red))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value

    title = 'Maximum place field size'
    xaxis = dict['xaxis']
    labels = ['Dimension', 'Max PF size'] 
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath, ynormalized=False)

elif plot == 'meansize':

    allowed_keys = []
    for red in red_vect:
        allowed_keys.append("redun = " + str(red))
    dict2 = {}
    for key, value in dict.items():
        if key in allowed_keys:
            dict2[key] = value

    title = 'Mean place field size'
    xaxis = dict['xaxis']
    labels = ['Dimension', 'Mean PF size'] 
    print("Plotting results...")
    plot_errorplot(dict2, xaxis, title, labels, basepath, ynormalized=False)