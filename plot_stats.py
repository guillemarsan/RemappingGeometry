import pathlib, argparse, json, time
import numpy as np

from convexsnn.plot import plot_errorplot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--num_dimensions", type=int, default=4,
                        help="Number of dimensions")
    parser.add_argument("--num_redundancy", type=int, default=7,
                        help="Number of redundancy levels")                   
    parser.add_argument("--num_dirs", type=int, default=4,
                        help="Number of input directions")
    parser.add_argument("--num_loadid", type=int, default=3,
                        help="Number of bbox loadids used")
    parser.add_argument('--dir_vect', nargs='+', type=int, default=[0],
                        help="Direction of the input vector")
    parser.add_argument('--loadid_vect', nargs='+', type=int, default=[0],
                        help="LoadID of the bbox vector")
    parser.add_argument("--read_dir", type=str, default='./data/',
                        help="Directory to read files")
    parser.add_argument("--write_dir", type=str, default='./out/',
                        help="Directory to dump output")

    args = parser.parse_args()

timestr = time.strftime("%Y%m%d-%H%M%S")
name = "pcs-" + str(args.dim_pcs)
basepath = args.write_dir + timestr + "-" + name

red_vect = 2**(np.arange(args.num_redundancy))
dim_vect = 2**(np.arange(args.num_dimensions)+2)
if args.num_dirs != 0:
    dir_vect = np.arange(args.num_dirs)
else:
    dir_vect = np.array(args.dir_vect)
num_dirs = dir_vect.shape[0]
if args.num_loadid != 0:
    loadid_vect = np.arange(args.num_loadid)
    num_loadid = args.num_loadid
else:
    loadid_vect = np.array(args.loadid_vect)
num_loadid = loadid_vect.shape[0]


patt = "*.json"
path = pathlib.Path(args.read_dir + "RandTorusPCS" + str(args.dim_pcs))
results_files = path.rglob(patt)
results = np.zeros((args.num_redundancy,args.num_dimensions,num_dirs, num_loadid))

for f in results_files:
    with open(f) as res_file:
        f = json.load(res_file)
        n = f['args']['nb_neurons']
        d = f['args']['dim_bbox']
        dir = int(f['args']['input_dir'][0])
        loadid = f['args']['load_id']

        dim_idx = np.argwhere(dim_vect == d)[0,0]
        red_idx = np.argwhere(red_vect == n/d)[0,0]
        if dir in dir_vect: dir_idx = np.argwhere(dir_vect == dir)[0,0]
        if loadid in loadid_vect: loadid_idx = np.argwhere(loadid_vect == loadid)[0,0]

        if dir in dir_vect and loadid in loadid_vect:
            results[red_idx, dim_idx, dir_idx, loadid_idx] = f['pcs_percentage']

dict = {}
for r in np.arange(args.num_redundancy):
    key = 'redun = ' + str(red_vect[r])
    dict[key] = results[r,:,:].reshape(args.num_dimensions, num_dirs*num_loadid)

title = 'Percentage of place cells for dim_pcs = ' + str(args.dim_pcs)
xaxis = dim_vect
labels = ['Dimension', 'Percentage of place cells']
plot_errorplot(dict, xaxis, title, labels, basepath)