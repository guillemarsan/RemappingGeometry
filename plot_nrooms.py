import pathlib, argparse, json, time
import numpy as np

from convexsnn.plot import plot_errorplot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--num_dimensions", type=int, default=4,
                        help="Number of dimensions")
    parser.add_argument("--num_redundancy", type=int, default=2,
                        help="Number of redundancy levels")                   
    parser.add_argument("--num_dirs", type=int, default=6,
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

dir_vect = np.arange(args.num_dirs)
loadid_vect = np.arange(args.num_loadid)  
num_loadid = loadid_vect.shape[0]

red_vect = 2**(np.arange(args.num_redundancy)+5)
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

dict = {}
nbins=np.arange(args.num_dirs+2)

for r in red_vect:
    for d in dim_vect:
        n = r*d
        patt = "*-bbox-" + str(d) + "-n-" + str(n) + ".json"
        path = pathlib.Path(args.read_dir + "RandTorusPCS" + str(args.dim_pcs))
        results_files = path.rglob(patt)
        results = np.zeros((n,args.num_dirs, num_loadid))

        for f in results_files:
            with open(f) as res_file:
                f = json.load(res_file)
                
                dir = int(f['args']['input_dir'][0])
                loadid = f['args']['load_id']

                dir_idx = np.argwhere(dir_vect == dir)[0,0]
                loadid_idx = np.argwhere(loadid_vect == loadid)[0,0]

                active = np.array(f['pcs'])
                results[active, dir_idx, loadid_idx] = 1

        key = 'b,n = ' + str(d) + "," + str(n)
        nrooms = np.sum(results[:,:,:],axis=2)
        hist = np.apply_along_axis(lambda a: np.histogram(a, bins=nbins)[0], 0, nrooms)
        dict[key] = hist/n

title = 'Percentage of neurons active in n rooms'
xaxis = nbins[:-1]
labels = ['Number of rooms', 'Percentage of PCs']
plot_errorplot(dict, xaxis, title, labels, basepath)