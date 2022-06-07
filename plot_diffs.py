import pathlib, argparse, json, time
import numpy as np

import matplotlib.pyplot as plt

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


plt.figure("f1")
plt.figure("f2")
plt.figure("f3")
color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
color_idx = 0
for r in red_vect:
    for d in dim_vect:
        n = r*d
        patt = "*-bbox-" + str(d) + "-n-" + str(n) + ".json"
        path = pathlib.Path(args.read_dir + "RandTorusPCS" + str(args.dim_pcs))
        results_files = path.rglob(patt)
        results = np.zeros((n,args.num_dirs, num_loadid))
        results_Theta = np.zeros((d,4,args.num_dirs, num_loadid))

        for f in results_files:
            with open(f) as res_file:
                f = json.load(res_file)
                
                dir = int(f['args']['input_dir'][0])
                loadid = f['args']['load_id']

                dir_idx = np.argwhere(dir_vect == dir)[0,0]
                loadid_idx = np.argwhere(loadid_vect == loadid)[0,0]

                active = np.array(f['pcs'])
                results[active, dir_idx, loadid_idx] = 1
                results_Theta[:,:,dir_idx,loadid_idx] = np.array(f['basis'])

        px_vect = np.array([])
        py_vect = np.array([])
        for b in np.arange(args.num_dirs):
            for c in np.arange(b):
                # Other measures
                # px = np.linalg.norm(np.linalg.norm(results_Theta[:,:,b,0] - results_Theta[:,:,c,0],axis=0),axis=0)/d**2
                # div = np.sum(np.sign(results[:,b,:] + results[:,c,:]),axis=0)
                # py = np.mean(np.linalg.norm(results[:,b,:] - results[:,c,:],axis=0)/div)

                divpx = np.linalg.norm(results_Theta[:,:,b,0])*np.linalg.norm(results_Theta[:,:,c,0])
                px = np.sum(results_Theta[:,:,b,0] * results_Theta[:,:,c,0])/divpx
                divpy = np.linalg.norm(results[:,b,:])*np.linalg.norm(results[:,c,:])
                py = np.mean(np.sum(results[:,b,:] * results[:,c,:],axis=0)/divpy)

                px_vect = np.append(px_vect, px)
                py_vect = np.append(py_vect,py)
        
        plt.figure("f1")
        label = 'b,n = ' + str(d) + "," + str(n)
        plt.scatter(px_vect, py_vect, color=color_list[color_idx], label=label)
        coeffs = np.polyfit(px_vect,py_vect,1)
        samples = np.linspace(np.min(px_vect)-0.01,np.max(px_vect)+0.01,100)
        pyhat = coeffs[1] + coeffs[0]*samples
        plt.plot(samples,pyhat,color=color_list[color_idx])

        
        nbins_pcs = np.arange(0,1,0.02)
        nbins_embedds = np.arange(-1,1,0.02)
        npts = np.math.factorial(args.num_dirs)/(2*np.math.factorial(args.num_dirs-2))
        hist_pcs = np.histogram(py_vect, bins=nbins_pcs)[0]/npts
        hist_embedds = np.histogram(px_vect, bins=nbins_embedds)[0]/npts
        plt.figure("f2")
        plt.step(nbins_pcs[:-1], hist_pcs, color=color_list[color_idx],where="post",linewidth=1,label=label)
        plt.figure("f3")
        plt.step(nbins_embedds[:-1], hist_embedds, color=color_list[color_idx],where="post",linewidth=1,label=label)

        color_idx += 1

plt.figure("f1")
plt.title('Relationship between angle in embeddings and active cells')
plt.xlabel('Angle in embedding')
plt.ylabel('Angle in active cells')
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
filepath = "%s-diffs.png" % basepath
plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.figure("f2")
plt.title('Frequency of angles in pcs')
plt.xlabel('Angle in active cells')
plt.ylabel('Probability')
plt.ylim([0,1])
plt.xlim(left=0)
plt.legend()
plt.tight_layout()
filepath = "%s-freq_diff_pcs.png" % basepath
plt.savefig(filepath, dpi=600, bbox_inches='tight')

plt.figure("f3")
plt.title('Frequency of angles in embeddings')
plt.xlabel('Angle in embeddings')
plt.ylabel('Probability')
plt.ylim([0,1])
plt.xlim(left=0)
plt.legend()
plt.tight_layout()
filepath = "%s-freq_diff_embdds.png" % basepath
plt.savefig(filepath, dpi=600, bbox_inches='tight')

