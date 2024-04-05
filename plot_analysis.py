from operator import ne
import pathlib, argparse, json, time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats.stats import pearsonr
from scipy import ndimage
import numpy as np
import time

from scipy.spatial import ConvexHull
from convexsnn.AngleEncoder import AngleEncoder
from convexsnn.plot import plot_scatterplot, plot_violinplot, plot_displot, plot_errorplot_cond


############# PLOTTING ##########################################

def pearson(x,y,i):

    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    coeffs = np.polyfit(x,y,1)
    Rp = pearsonr(x, y)
    formatstr = "r = {:.4f} \n" + ("p = {:.3e}" if Rp[1] < 0.001 else "p = {:.3f}")
    plt.text(0.1, 0.9-0.1*i, formatstr.format(Rp[0], Rp[1]), transform=plt.gca().transAxes)
    samples = np.linspace(np.min(x)-0.01,np.max(x)+0.01,100)
    pyhat = coeffs[1] + coeffs[0]*samples
    plt.plot(samples,pyhat,color=c[i])

def compute_mesh(D):
    x, y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    mesh = np.vstack([x.ravel(), y.ravel()])
    Encoder = AngleEncoder()
    kgrid = Encoder.encode(mesh)
    rates = D.T @ kgrid
    argmaxs = np.argmax(rates, axis=0)
    return mesh, argmaxs

def make_plot(ptype, data, title, axis_labels, basepath, legends=None, ynormalized=False, equal=False, flipxaxis=False, colorsvect=None):

    plt.figure(figsize=(4,4))
    ax = plt.gca()

    if ptype == 'line': 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0
        ci = -1
        for _, point in data.iterrows():
            mean = point['mean']
            err = point['sem']
            xaxis = point['xaxis']
            legend = point['legend'] if legends is None else legends[i]

            if legend.endswith('shuff') or legend.endswith('(Shuffle)'):
                plt.errorbar(xaxis, mean, err, marker='.', linestyle='--', capsize=3, label=legend, color=c[ci])
            else:
                ci += 1
                plt.errorbar(xaxis, mean, err, marker='.', capsize=3, label=legend, color=c[ci])
                
            i += 1

    elif ptype == 'distr':
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        xaxis = np.array(data['xaxis'])
        scatter = False
        j = 0
        for column in data:
            if column not in {'xaxis','legend'}:
                legend = column
                data_class = np.array(data[column])
                medians = np.array(list(map(lambda x: np.quantile(x, 0.5), data_class)))
                lowquartiles = np.array(list(map(lambda x: np.quantile(x, 0.25), data_class)))
                highquartiles = np.array(list(map(lambda x: np.quantile(x, 0.75), data_class)))
                err = np.array(list(zip(lowquartiles, highquartiles))).T
                plt.errorbar(xaxis, medians, yerr=err, marker='.', capsize=3, label=legend, color=c[j])
                if scatter:
                    i = 0
                    for _, points in data_class.iteritems():
                        xpoints = np.ones(points.shape[0])*xaxis[i]
                        plt.scatter(xpoints, points, color='w', edgecolor=c[j], s=5, zorder=20)
                        i += 1
                j += 1


    elif ptype == 'hist':
        for _, point in data.iterrows():
            hist = point['hist']
            bins = point['bins']
            legend = point['legend']

            widths = np.diff(bins)
            xticks = bins[:-1] + widths/2
            plt.bar(xticks, hist, width=widths, align='center', label=legend)

    elif ptype == 'bars_shuff':
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        conditions = []

        xticks = np.arange(len(data),step=2)
        j = 0
        for i in xticks:
            
            point = data.iloc[i]
            pointshuff = data.iloc[i+1]
            mean = point['mean']
            sem = point['sem']
            meanshuff = pointshuff['mean']
            semshuff = pointshuff['sem']
            values = point['values'][0]

            
            conditions.append(point['xaxis'][0])
            
            plt.bar(j, mean, width=0.7, align='center', color=c[i], alpha=0.5)
            jitter = 0 #np.random.uniform(-0.05,0.05,values.shape[0])
            plt.scatter(np.ones_like(values)*j+jitter, values, color=c[i], edgecolor='grey', linewidth=0.5)
            plt.hlines(meanshuff, j-0.4, j+0.4, color=c[i])
            plt.errorbar(j, mean, 2*sem[0], ecolor='black',capsize=10)
            j += 1
            
        plt.xticks(np.arange(j), conditions)
        plt.gca().set_aspect(3)

    elif ptype == 'bars':
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        conditions = []

        ps = len(data)
        iticks = np.arange(len(data.iloc[0]['xaxis']))
        jticks = np.linspace(0,0.4,ps) - 0.2
        pad = np.abs(jticks[0]-jticks[1])-0.05
        j = 0
        for _,point in data.iterrows():
            
            means = point['mean']
            sems = point['sem']
            xaxis = point['xaxis']

            for i in iticks:
                plt.bar(i+jticks[j], means[i], width=pad, align='center', color=c[j])
                plt.errorbar(i+jticks[j], means[i], sems[i], ecolor='black',capsize=50*pad)
            j += 1

        for i in iticks:
            conditions.append(xaxis[i])
        plt.xticks(iticks, conditions)

    elif ptype == 'scatter':
        

        for _, point in data.iterrows():
            data = point['data']
            datashuff = point['datashuff']

            
            plt.scatter(data[:,0],data[:,1], alpha=0.7, label=point['legend'])
            pearson(data[:,0], data[:,1], 0)
            plt.scatter(datashuff[:,0], datashuff[:,1], alpha=0.7, label='shuffle')
            pearson(datashuff[:,0], datashuff[:,1], 1)



        if ynormalized: plt.xlim(0,1.1)

    elif ptype == 'pfs2':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if colorsvect is not None: 
            colorgroups = np.concatenate([np.full(count, i) for i, count in enumerate(colorsvect)])
        cases = len(data)
        neurons = data.iloc[0]['neurons']
        num_neurons = len(neurons)
        i = 1
        
        for _, point in data.iterrows():
            pfs = point['pfs']

            axsub = plt.subplot(1, cases, i)
            axsub.set_title(point['legend'], fontsize=7)
            axsub.set_xticks([])
            axsub.set_yticks([])
            ti = 1

            maxFR = np.max(pfs)
            for cell in np.arange(num_neurons):

                c_sat = c[cell % 10] if colorsvect is None else c[colorgroups[cell % 10]]
                cmap = colors.ListedColormap(['white', c_sat])
                bounds=[0,0,maxFR]
                norm = colors.BoundaryNorm(bounds, cmap.N)

                pf = pfs[cell,:]
                sqrtb = int(np.sqrt(pfs.shape[1]))
                pf = pf.reshape((sqrtb,sqrtb)).T
                alphas = pf/np.max(pf) if np.max(pf) > 0 else 0
                im = axsub.imshow(pf, alpha=alphas, cmap=cmap, norm=norm, extent=[-1,1,-1,1])
                if np.max(pf)>0:
                    axsub.text(0.7, -ti*0.15, '{:.2f}Hz'.format(np.max(pf)), wrap=True, transform=axsub.transAxes, color=c_sat, horizontalalignment='center', fontsize=10)
                    ti += 1
            i += 1

        if num_neurons < 20:
            if colorsvect is None:
                patches = [matplotlib.patches.Patch(color=c[i % 10], label='N%i' % n) for i,n in enumerate(neurons) ]
            else:
                patches = [matplotlib.patches.Patch(color=c[colorgroups[i]], label='N%i' % n) for i,n in enumerate(neurons) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , prop={'size': 6})
       
    elif ptype == 'pfs1':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if colorsvect is not None: 
            colorgroups = np.concatenate([np.full(count, i) for i, count in enumerate(colorsvect)])
        cases = len(data)
        neurons = data.iloc[0]['neurons']
        num_neurons = len(neurons)

        i = 1
        for _, point in data.iterrows():
            pfs = point['pfs']

            axsub = plt.subplot(cases, 1, i)
            axsub.set_title(point['legend'], fontsize=7)
            for cell in np.arange(num_neurons):
                pf = pfs[cell,:]
                axsub.plot(pf, color=c[cell % 10] if colorsvect is None else c[colorgroups[cell % 10]])
            i += 1
        
        if num_neurons < 20:
            if colorsvect is None:
                patches = [matplotlib.patches.Patch(color=c[i % 10], label='N%i' % n) for i,n in enumerate(neurons) ]
            else:
                patches = [matplotlib.patches.Patch(color=c[colorgroups[i]], label='N%i' % n) for i,n in enumerate(neurons) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , prop={'size': 6})

    elif ptype == 'pca':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.figure(figsize=(10,10))
        point = data.iloc[0]
        var_r = point['var_r']
        proj_r = point['proj_r']

        ax1 = plt.subplot(1, 2, 1, projection='3d')
        for e in np.arange(proj_r.shape[2]):
            ax1.scatter(proj_r[0,:,e], proj_r[1,:,e], proj_r[2,:,e])
        ax1.set_title('PCA of r')

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(var_r.shape[0]), var_r)
        ax2.set_title('Var. explained %')
        ax2.set_xlabel('PC')

    elif ptype == 'vis':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.figure(figsize=(10,10))
        point = data.iloc[0]
        x = point['x_trajs']
        xhat = point['xhat_trajs']
        g = point['g_trajs']
        ghat = point['ghat_trajs']

        cylinder = True
        hat = True
        if x.shape[0] == 3 or cylinder:
            ax3 = plt.subplot(1, 2, 1, projection='3d')
            for e in np.arange(x.shape[2]):
                ax3.plot(x[0,:,e], x[1,:,e], x[2,:,e])
                ax3.scatter(x[0,0,e], x[1,0,e], x[2,0,e], color=c[e], s=150)
                if hat: ax3.plot(xhat[0,:,e], xhat[1,:,e], xhat[2,:,e], color = c[e], alpha=0.5)
            ax3.set_title('Embedding space y')
            ax3.set_xlim([-np.sqrt(2),np.sqrt(2)])
            ax3.set_ylim([-np.sqrt(2),np.sqrt(2)])
            if not cylinder: 
                ax3.set_zlim([-1,1])
            ax3.set_box_aspect([np.ptp(axis) for axis in [ax3.get_xlim(), ax3.get_ylim(), ax3.get_zlim()]])
        else:
            ax3a = plt.subplot(2, 2, 1)
            for e in np.arange(proj_r.shape[2]):
                ax3a.plot(x[0,:,e], x[1,:,e])
                if hat: ax3a.plot(xhat[0,:,e], xhat[1,:,e])
            ax3a.set_xlim([-1.1,1.1])
            ax3a.set_ylim([-1.1,1.1])
            ax3a.set_title('y(0),y(1)')

            ax3b = plt.subplot(2, 2, 2)
            for e in np.arange(proj_r.shape[2]):
                ax3b.scatter(x[2,:,e], x[3,:,e])
                if hat: ax3a.plot(xhat[2,:,e], xhat[3,:,e])
            ax3b.set_xlim([-1.1,1.1])
            ax3b.set_ylim([-1.1,1.1])
            ax3b.set_title('y(2),y(3)')      

        fields = True
        if fields:
            D = point['D']
            mesh, argmax = compute_mesh(D)

        ax4 = plt.subplot(1, 2, 2)
        if fields:
            for i in np.arange(D.shape[1]):
                meshi = mesh.copy()
                meshi[:, argmax != i] = 0
                dims = int(np.sqrt(meshi.shape[1]))
                map2d = np.ones((meshi.shape[1]))
                map2d[np.argwhere(argmax != i)] = 0
                map2d = map2d.reshape((dims,dims))
                structure = np.ones((3, 3), dtype=int)

                labeled, ncomponents = ndimage.label(map2d, structure)
                labeled = labeled.reshape(-1)
                for com in np.arange(ncomponents) + 1:
                    com_lab = np.argwhere(labeled == com)[:,0]
                    if com_lab.shape[0] > 2:
                        pointsi = meshi[:, com_lab]
                        if len(np.unique(pointsi[0,:])) > 1 and len(np.unique(pointsi[1,:])) > 1:
                            hullni = ConvexHull(pointsi.T)
                            ax4.fill(pointsi[0,hullni.vertices], pointsi[1,hullni.vertices], alpha=0.5, color=c[i%10], lw=0)

        for e in np.arange(g.shape[2]):
            disc_idx = np.where(np.linalg.norm(np.diff(g[:,:,e]),axis=0) > 0.5)[0]+1
            disc_g = np.hsplit(g[:,:,e], disc_idx)
            for g_seg in disc_g:
                ax4.plot(g_seg[0,:], g_seg[1,:], color=c[e])
            ax4.scatter(g[0,0,e], g[1,0,e], color=c[e])
            if hat:
                disc_idx = np.where(np.linalg.norm(np.diff(ghat[:,:,e]),axis=0) > 0.5)[0]+1
                disc_ghat = np.hsplit(ghat[:,:,e], disc_idx)
                for ghat_seg in disc_ghat:
                    ax4.plot(ghat_seg[0,:], ghat_seg[1,:], color=c[e], alpha=0.5)
        ax4.set_title('Instrinsic manifold')
        ax4.set_xlim([-1,1])
        ax4.set_ylim([-1,1])
        ax4.set_aspect('equal')  
    
    if ptype not in {'pfs2','pfs1','pca', 'vis'}:
        plt.title(title, fontsize=10)
        plt.xlabel(axis_labels[0], fontsize=10)
        plt.ylabel(axis_labels[1], fontsize=10)
        if ynormalized: plt.ylim(0,1.1)
        if equal: plt.gca().set_aspect('equal')
        if flipxaxis: plt.gca().invert_xaxis()    
        plt.tick_params(axis='both', labelsize=10)
        plt.legend(frameon=False, prop={'size': 10})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

    filepath = "{0}-{1}_plot.svg".format(basepath,ptype)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


############# LOADING AND PREPARING DATA ########################
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

def filter(df, visualize, params_sweep, tags):

    full_keys = []
    full_keys.extend(visualize)
    for t in tags:
        full_keys.append(t)

    filteredf = pd.DataFrame()
    for tuple in params_sweep:
        points = df[(df[visualize] == tuple).all(1)]
        if len(points.index) > 0:
            filteredf = pd.concat([filteredf, points[full_keys]])
    return filteredf

def prepare_single(df, labelsin, labelsout, labelsparams, visualize):

    newdf = pd.DataFrame([])
    for _, point in df.iterrows():
        newpoint = {}
        for i in np.arange(len(labelsin)):
            value = point[labelsin[i]]
            if type(value) is str and value.startswith('['):
                newpoint[labelsout[i]] = [np.array(eval(point[labelsin[i]]))]
            else:
                newpoint[labelsout[i]] = value
        legend = ''
        i = 0
        for l_idx, l in enumerate(labelsparams):
            legend += l + '=' + str(point[visualize[i]])
            legend += ', ' if l_idx != len(labelsparams)-1 else ''
            i += 1
        newpoint['legend'] = legend
        newdf = pd.concat([newdf, pd.DataFrame(newpoint)])
    return newdf  

def prepare_combine(df, across, data_label, xaxis_label, labels, visualize):
    def combinelambda(set):
        points = pd.DataFrame([])
        for dlabel in data_label:
            newpoint = {}
            list = set[dlabel]
            list = list.apply(lambda x: np.array(eval(x)))

            if across:
                means = np.mean(list._values)
                sems = np.std(list._values)/np.sqrt(len(list))
                xaxis = np.array(eval(set.iloc[0][xaxis_label]))
            else:
                means = []
                sems = []
                xaxis = []
                i = 0
                for elem in list._values:
                    means.append(np.mean(elem))
                    sems.append(np.std(elem)/np.sqrt(elem.shape[0]))
                    xaxis.append(set.iloc[i][xaxis_label])
                    i += 1
                newpoint['values'] = [list._values]
            newpoint['mean'] = [means]
            newpoint['sem'] = [sems]
            newpoint['xaxis'] = [xaxis]

            legend = ''
            i = 0
            for l_idx, l in enumerate(labels):
                legend += l + '=' + str(set.iloc[0][visualize[i]])
                legend += ', ' if l_idx != len(labels)-1 else ''
                i += 1
            legend += (', ' + dlabel) if len(data_label) > 1 else ''
            newpoint['legend'] = legend

            points = pd.concat([points, pd.DataFrame(newpoint)])
        return points

    # Mean and std within points and combine
    gb= df.groupby(visualize, as_index=False)
    newdf = gb.apply(lambda x: combinelambda(x))
    return newdf 

def prepare_pfs(df, neurons, visualize, labels, active=False):

    # Read the pfs
    newdf = pd.DataFrame([])
    i = 0
    all_pfs = []
    if active: df = df.iloc[:2]
    for _, row in df.iterrows():
        with open(row['pfs']) as res_file:
            t0 = time.time()
            pfs = np.genfromtxt(res_file)
            t1 = time.time()
            print(t1-t0)
        all_pfs.append(pfs)

    all_pfs = np.array(all_pfs)
    if not active:
        array_n = neurons
    else:
        array_n = []
        active1 = np.any(all_pfs[0,:,:], axis=1)
        active2 = np.any(all_pfs[1,:,:], axis=1)
        array_n.append(np.argwhere(active1 * active2)[:6][:,0][:])
        array_n.append(np.argwhere(active1 * ~active2)[:2][:,0])
        array_n.append(np.argwhere(active2 * ~active1)[:2][:,0])
        array_n = np.hstack(array_n)

    for i in range(all_pfs.shape[0]):
        newpoint = row[visualize]
        newpoint['pfs'] = all_pfs[i, array_n, :]
        newpoint['neurons'] = array_n

        legend = ''
        j = 0
        for l_idx, l in enumerate(labels):
            legend += l + '=' + str(row[visualize[j]])
            legend += ', ' if l_idx != len(labels)-1 else ''
            j += 1
        newpoint['legend'] = legend

        newdf = pd.concat([newdf, pd.DataFrame([newpoint])])

    return newdf


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dir", type=str, default='gridcellsd1M',
                        help="Directory to read and write files")
    parser.add_argument("--dir_loc", type=str, default='./data/v1',
                        help="Location of the directory" )
    parser.add_argument("--plot", type=str, default='vis',
                        help = 'Which plot to make')
    

    args = parser.parse_args()

plot = args.plot

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = args.dir_loc + '/' + args.dir + '/'
name = basepath +  timestr + '-' + args.dir
    
print("Loading results...")

# Load DataFrame
if plot in {'placefields'}:
    df = load_dataframe('database', basepath)
elif plot in {'spatialinfo'}:
    df = load_dataframe('placecells', basepath)
elif plot in {'remapping','nrooms', 'measures', 'variance_remap', 'dims_remap', 'dims_stats', 'multispatialcorr', 
            'frdistance_pertuning', 'spatialcorr_pertuning', 'nrooms_pertuning', 'pca', 'remap_vec', 'vis'}:
    df = load_dataframe('remapping', basepath)
else:
    df = load_dataframe('recruitment', basepath)

# Plotting
print ("Plotting results...")

#### General plots ################

if plot == 'placefields':

    dim_pcs = df.iloc[0]['arg_dim_pcs']
    n_neurons = df.iloc[0]['arg_nb_neurons']
    active = dim_pcs == 2
    visualize = ['arg_path_type','arg_simulate']
    labels = ['p', 's']
    params_sweep = [('grid', 'minimization')]
    tags = ['pfs']
    neurons = np.arange(n_neurons) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colorsvect = None #[1,1,1,1,1]]
    

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_pfs(df, neurons, visualize, labels, active=active)

    title = ''
    axis_labels = []
    print("Plotting results...")
    name += '-' + plot
    make_plot('pfs2' if dim_pcs == 2 else 'pfs1', newdf, title, axis_labels, name, colorsvect=colorsvect)


#### Place cell plots #############
elif plot == 'spatialinfo':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id', 'arg_encoding']
    labels = ['d', 'e', 'n', 'l']
    params_sweep = [(4,2,8,0,'rotation')]
    tags = ['spatialinfo', 'spatialinfo_bins']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['hist','bins'], labels, visualize)
    

    title = 'Spatial information distribution'
    axis_labels = ['Spatial information (bits/s)', 'Percentage of neurons']
    print("Plotting results...")
    name += '-' + plot
    make_plot('hist', newdf, title, axis_labels, name, ynormalized=False)

#### Remapping plots #############
if plot == 'remapping' or plot == 'nrooms':

    visualize = ['arg_dim_bbox','arg_nb_neurons']
    labels = ['d', 'n']
    params_sweep = [(64,1024)]
    tags = ['nrooms', 'nrooms_bins']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=True, data_label=['nrooms'], xaxis_label='nrooms_bins', labels=labels, visualize=visualize)

    title = 'Percentage of neurons active in n rooms'
    axis_labels = ['Number of rooms', 'Percentage of PCs']
    print("Plotting results...")
    namea = name + '-' + 'nrooms'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False)

if plot == 'remapping' or plot == 'measures':

    visualize = ['arg_path_type','arg_simulate']
    labels = ['p', 's']
    params_sweep = [('grid', 'minimization')]
    tags = ['overlap', 'overlapshuff', 'spatialcorr', 'spatialcorrshuff']
    
    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=False, data_label = tags, xaxis_label= 'arg_simulate', labels= labels, visualize=visualize)
    
    title = 'FR overlap and spatial corr. across environments'
    axis_labels = ['', 'Overlap and Spatial Corr.'] 
    print("Plotting results...")
    namea = name + '-' + 'measures'
    make_plot('bars_shuff', newdf, title, axis_labels, namea, ynormalized=True)

if plot == 'remapping' or plot == 'variance_remap':

    visualize = ['arg_encoding', 'arg_dim_bbox', 'arg_embedding_sigma']
    labels = ['e', 'd', 's']
    params_sweep = [('rotation',16,s) for s in [0, 0.1, 0.25, 0.5, 1]]
    tags = ['overlap', 'overlapshuff', 'spatialcorr', 'spatialcorrshuff']

    fdf = filter(df, visualize, params_sweep, tags)

    # Xaxis variable
    xaxis = 'arg_embedding_sigma'
    xaxislabel = 's'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labels, visualize=visualize)
    
    title = 'Environment dissimilarity'
    axis_labels = ['Dissimilarity', 'Metric'] 
    legends = ['Overlap', 'Overlap (Shuffle)', 'Spatial correlation', 'Spatial correlation (Shuffle)']
    print("Plotting results...")
    namea = name + '-' + 'variance_remap'
    make_plot('line', newdf, title, axis_labels, namea, legends, ynormalized=True)

if plot == 'remapping' or plot == 'dims_remap':

    visualize = ['arg_dim_bbox', 'arg_embedding_sigma']
    labels = ['d', 's']
    params_sweep = [(d, -1) for d in [8, 16, 32, 64, 128]]
    tags = ['overlap', 'overlapshuff', 'spatialcorr', 'spatialcorrshuff', 'overlapbin', 'overlapbinshuff']

    fdf = filter(df, visualize, params_sweep, tags)

    # Xaxis variable
    xaxis = 'arg_dim_bbox'
    xaxislabel = 'd'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labels, visualize=visualize)
    
    title = 'Embedding space dimensionality'
    axis_labels = ['Dimension', 'Metric'] 
    legends = ['Overlap', 'Overlap (Shuffle)', 'Spatial correlation', 'Spatial correlation (Shuffle)', 'Binary Overlap','Binary Overlap (Shuffle)']
    print("Plotting results...")
    namea = name + '-' + 'dims_remap'
    make_plot('line', newdf, title, axis_labels, namea, legends, ynormalized=True)

if plot == 'remapping' or plot == 'dims_stats':

    visualize = ['arg_dim_bbox', 'arg_embedding_sigma']
    labels = ['d', 's']
    params_sweep = [(d, -1) for d in   [8, 16, 32, 64, 128]]
    tags = ['perpcs', 'meanperpfsizes', 'dimp_errors', 'dime_errors'] # 'dimg_errors'] # 

    fdf = filter(df, visualize, params_sweep, tags)

    # Xaxis variable
    xaxis = 'arg_dim_bbox'
    xaxislabel = 'd'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labels, visualize=visualize)
    
    title = 'Statistics per dimension'
    axis_labels = ['Dimension', 'Metric'] 
    legends = ['PCs (%)', 'Mean PF size (%)', 'Position error', 'Env.variables error']
    print("Plotting results...")
    namea = name + '-' + 'dims_stats'
    make_plot('line', newdf, title, axis_labels, namea, legends, ynormalized=True)

if plot == 'remapping' or plot == 'multispatialcorr':

    visualize = ['arg_encoding', 'arg_dim_bbox']
    labels = ['e', 'd']
    params_sweep = [('rotation',8)]
    tags = ['multispatialcorr', 'multispatialcorrshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Pair of cells spatial correlation in two environements'
    axis_labels = ['Environment 1', 'Environment 2'] 
    print("Plotting results...")
    namea = name + '-' + 'multispatialcorr'
    make_plot('scatter', newdf, title, axis_labels, namea, ynormalized=True, equal=True)

if plot == 'remapping' or plot == 'frdistance_pertuning':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id', 'arg_encoding']
    labels = ['d', 'n', 'l']
    params_sweep = [(2,6,0, 'rotation')]
    tags = ['frdistance_pertuning', 'frdistance_pertuningshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Distance of FRs vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Distance of FRs'] 
    print("Plotting results...")
    namea = name + '-' + 'frdistance_pertuning'
    make_plot('scatter', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'spatialcorr_pertuning':

    visualize = ['arg_dim_bbox', 'arg_encoding', 'arg_embedding_sigma']
    labels = ['d', 'e', 's']
    params_sweep = [(64,'flexible',-1)]
    tags = ['spatialcorr_pertuning', 'spatialcorr_pertuningshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Spatial correlation vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Spatial correlation'] 
    print("Plotting results...")
    namea = name + '-' + 'spatialcorr_pertuning'
    make_plot('scatter', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'nrooms_pertuning':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id', 'arg_encoding']
    labels = ['d', 'n', 'l']
    params_sweep = [(2,6,0,'rotation')]
    tags = ['nrooms_pertuning', 'nrooms_pertuningshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Number of rooms vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Number of rooms'] 
    print("Plotting results...")
    namea = name + '-' + 'nrooms_pertuning'
    make_plot('scatter', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'pca':

    visualize = ['arg_path_type','arg_simulate']
    labels = ['p', 's']
    params_sweep = [('grid', 'minimization')]
    tags = ['var_explained_r', 'pca_proj_r']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['var_r','proj_r'], labels, visualize)
    
    title = 'PCA analysis'
    axis_labels = [] 
    print("Plotting results...")
    namea = name + '-' + 'pca'
    make_plot('pca', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'vis':

    visualize = ['arg_path_type','arg_simulate']
    labels = ['p', 's']
    params_sweep = [('grid', 'minimization')]
    tags = ['x_trajs', 'xhat_trajs', 'g_trajs', 'ghat_trajs', 'D']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags, ['x_trajs', 'xhat_trajs', 'g_trajs', 'ghat_trajs', 'D'], labels, visualize)
    
    title = 'Visualization tools'
    axis_labels = [] 
    print("Plotting results...")
    namea = name + '-' + 'vis'
    make_plot('vis', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'remap_vec':

    visualize = ['arg_path_type','arg_simulate']
    labels = ['p', 's']
    params_sweep = [('grid', 'minimization')]
    tags = ['norm_remap','norm_Dy','norm_Dye','norm_nu']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=False, data_label = tags, xaxis_label='arg_simulate', labels= labels, visualize=visualize)
    
    title = 'Remapping vector norm decompositon'
    axis_labels = ['','Norm'] 
    print("Plotting results...")
    namea = name + '-' + 'remap_vec'
    make_plot('bars', newdf, title, axis_labels, namea)

#### Recruitment plots #############

visualize = ['arg_tagging_sparse', 'arg_current_amp']
labels = ['S', 'C']

 # Xaxis variable
rec_type = 'total'
if rec_type == 'c':
    params_sweep = [(0, -500),(0.25, -500),(0.5,-500),(0.75,-500),(0.9,-500),(1.0,-500)]
    xaxis = 'arg_tagging_sparse'
    xaxislabel = 'S'
    invert = False
    xaxis_tag = 'Inhibition Sparseness S'
elif rec_type == 's':
    params_sweep = [(0.9, 0),(0.9, -100),(0.9,-500)]
    xaxis = 'arg_current_amp'
    xaxislabel = 'C'
    invert = True
    xaxis_tag = 'Inhibition Current C'
else:
    params_sweep = [(1.0, 0),(1.0, -500)]
    xaxis = 'arg_current_amp'
    xaxislabel = 'C'
    invert = False
    xaxis_tag = 'Inhibition Current C'

vis = visualize[:]
labs = labels[:]
vis.remove(xaxis)
labs.remove(xaxislabel)

if plot == 'recruitment' or plot == 'perpcs_perclass_line':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['perpcs_tP', 'perpcs_tM','perpcs_ntP','perpcs_ntM', 'perpcs_sP', 'perpcs_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Proportion of PCs in each class with Inhibition'
    axis_labels = [xaxis_tag, 'Proportion']
    print("Plotting results...")
    namea = name + '-' + 'perpcs_perclass_line'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'meanfr_perclass_line':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['meanfr_tP', 'meanfr_ntP', 'meanfr_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Mean FR per class'
    axis_labels = [xaxis_tag, 'Mean FR (Hz)',]
    print("Plotting results...")
    namea = name + '-' + 'meanfr_perclass_line'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'decoding_error':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['g_error']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Decoding error with Inhibition'
    axis_labels = [xaxis_tag, 'Decoding error']
    print("Plotting results...")
    namea = name + '-' + 'decoding_error'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'pcsmfr':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['pcsmeanfr']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'PCs FR with Inhibition'
    axis_labels = [xaxis_tag, 'FR (Hz)']
    print("Plotting results...")
    namea = name + '-' + 'pcsmfr'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'spikes':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['spikes']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Total number of spikes with Inhibition'
    axis_labels = [xaxis_tag, 'Number of spikes']
    print("Plotting results...")
    namea = name + '-' + 'spikes'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'spikes_perclass':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['spikes_tP', 'spikes_ntP','spikes_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Number of spikes per class with Inhibition'
    axis_labels = [xaxis_tag, 'Number of spikes']
    print("Plotting results...")
    namea = name + '-' + 'spikes_perclass'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'pcspfsize':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['pcspfsize']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Place field size with Inhibition'
    axis_labels = [xaxis_tag, 'Place field size (m2)']
    print("Plotting results...")
    namea = name + '-' + 'pcspfsize'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'meanfr_perclass':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['meanfr_tP', 'meanfr_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Mean FR per class'
    axis_labels = ['', 'Mean FR (Hz)']
    print("Plotting results...")
    namea = name + '-' + 'meanfr_perclass'
    make_plot('bars', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'perpcs_perclass':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['perpcs_tP', 'perpcs_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Proportion of PCs in each class with Inhibition'
    axis_labels = ['', 'Proportion']
    print("Plotting results...")
    namea = name + '-' + 'perpcs_perclass'
    make_plot('bars', newdf, title, axis_labels, namea, ynormalized=False, flipxaxis=invert)

if plot == 'recruitment' or plot == 'coveredarea_perclass':

    # params_sweep = [(16,0,512,0),(16,0,512,-20),(16,0,512,-40),(16,0,512,-60),(16,0,512,-80),(16,0,512,-100)]
    tags = ['coveredarea', 'coveredarea_tP', 'coveredarea_ntP', 'coveredarea_sR']

    fdf = filter(df, visualize, params_sweep, tags)

    newdf = prepare_combine(fdf, across=False, data_label=tags, xaxis_label=xaxis, labels=labs, visualize=vis)

    title = 'Covered area with Inhibition'
    axis_labels = [xaxis_tag, 'Area covered (%)']
    print("Plotting results...")
    namea = name + '-' + 'coveredarea_perclass'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=True, flipxaxis=invert)
