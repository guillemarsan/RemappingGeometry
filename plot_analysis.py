from operator import ne
import pathlib, argparse, json, time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats.stats import pearsonr
import numpy as np
import time

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
        for i in xticks:
            
            point = data.iloc[i]
            pointshuff = data.iloc[i+1]
            mean = point['mean']
            sem = point['sem']
            meanshuff = pointshuff['mean']
            semshuff = pointshuff['sem']

            
            conditions.append(point['xaxis'][0])
            
            j = i//2
            plt.bar(j-0.2, mean, width=0.36, align='center', color=c[i])
            plt.errorbar(j-0.2, mean, sem, ecolor='black',capsize=10)
            plt.bar(j+0.2, meanshuff, width=0.36, align='center', color=c[i],alpha=0.5)
            plt.errorbar(j+0.2, meanshuff, semshuff, ecolor='black',capsize=10)

        plt.xticks(xticks//2, conditions)

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

    elif ptype == 'pfs':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if colorsvect is not None: 
            colorgroups = np.concatenate([np.full(count, i) for i, count in enumerate(colorsvect)])
        cases = len(data)
        neurons = data.iloc[0]['neurons']
        num_neurons = neurons.shape[0]
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

                c_sat = c[cell] if colorsvect is None else c[colorgroups[cell]]
                cmap = colors.ListedColormap(['white', c_sat])
                bounds=[0,0,maxFR]
                norm = colors.BoundaryNorm(bounds, cmap.N)

                pf = pfs[cell,:]
                sqrtb = int(np.sqrt(pfs.shape[1]))
                pf = pf.reshape((sqrtb,sqrtb))
                alphas = pf/np.max(pf) if np.max(pf) > 0 else 0
                im = axsub.imshow(pf, alpha=alphas, cmap=cmap, norm=norm)
                if np.max(pf)>0:
                    axsub.text(0.7, -ti*0.15, '{:.2f}Hz'.format(np.max(pf)), wrap=True, transform=axsub.transAxes, color=c_sat, horizontalalignment='center', fontsize=4)
                    ti += 1
            i += 1

        if colorsvect is None:
            patches = [matplotlib.patches.Patch(color=c[i], label='N%i' % n) for i,n in enumerate(neurons) ]
        else:
            patches = [matplotlib.patches.Patch(color=c[colorgroups[i]], label='N%i' % n) for i,n in enumerate(neurons) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , prop={'size': 6})
    if ptype != 'pfs':
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
                sems = np.std(list._values)/len(list)
                xaxis = np.array(eval(set.iloc[0][xaxis_label]))
            else:
                means = []
                sems = []
                xaxis = []
                i = 0
                for elem in list._values:
                    means.append(np.mean(elem))
                    sems.append(np.std(elem)/elem.shape[0])
                    xaxis.append(set.iloc[i][xaxis_label])
                    i += 1
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

def prepare_pfs(df, neurons, visualize, labels):

    # Read the pfs
    newdf = pd.DataFrame([])
    i = 0
    for _, row in df.iterrows():
        newpoint = row[visualize]
        with open(row['pfs']) as res_file:
            t0 = time.time()
            pfs = np.genfromtxt(res_file)
            t1 = time.time()
            print(t1-t0)
        newpoint['pfs'] = pfs[neurons, :]
        newpoint['neurons'] = neurons
       
        legend = ''
        j = 0
        for l_idx, l in enumerate(labels):
            legend += l + '=' + str(row[visualize[j]])
            legend += ', ' if l_idx != len(labels)-1 else ''
            j += 1
        newpoint['legend'] = legend

        newdf = pd.concat([newdf, pd.DataFrame([newpoint])])
        i += 1

    return newdf


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dir", type=str, default='TestRecruitmentAlmost',
                        help="Directory to read and write files")
    parser.add_argument("--plot", type=str, default='remapping',
                        help = 'Which plot to make')
    

    args = parser.parse_args()

plot = args.plot

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = './data/' + args.dir + '/'
name = basepath +  timestr + '-' + args.dir
    
print("Loading results...")

# Load DataFrame
if plot in {'placefields'}:
    df = load_dataframe('database', basepath)
elif plot in {'spatialinfo'}:
    df = load_dataframe('placecells', basepath)
elif plot in {'remapping','nrooms', 'overlap', 'spatialcorr', 'variance_remap', 'multispatialcorr', 
            'frdistance_pertuning', 'spatialcorr_pertuning', 'nrooms_pertuning'}:
    df = load_dataframe('remapping', basepath)
else:
    df = load_dataframe('recruitment', basepath)

# Plotting
print ("Plotting results...")

#### General plots ################

if plot == 'placefields':

    visualize = ['arg_current_amp']
    labels = ['C']
    params_sweep = [(0), (-10), (-20), (-30), (-100)]
    tags = ['pfs']
    neurons = np.array([47,0,2,292,16,28,8,3,7,14,20,25,35])
    colorsvect = [3,3,3,3,1]

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_pfs(df, neurons, visualize, labels)

    title = ''
    axis_labels = []
    print("Plotting results...")
    name += '-' + plot
    make_plot('pfs', newdf, title, axis_labels, name, colorsvect=colorsvect)



#### Place cell plots #############
elif plot == 'spatialinfo':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id', 'arg_encoding']
    labels = ['d', 'e', 'n', 'l']
    params_sweep = [(8,0,256,0,'rotation')]
    tags = ['spatialinfo', 'spatialinfo_bins']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(df, tags,['hist','bins'], labels, visualize)
    

    title = 'Spatial information distribution'
    axis_labels = ['Spatial information (bits/s)', 'Percentage of neurons']
    print("Plotting results...")
    name += '-' + plot
    make_plot('hist', newdf, title, axis_labels, name, ynormalized=False)

#### Remapping plots #############
if plot == 'remapping' or plot == 'nrooms':

    visualize = ['arg_dim_bbox','arg_nb_neurons']
    labels = ['d', 'n']
    params_sweep = [(16,512)]
    tags = ['nrooms', 'nrooms_bins']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=True, data_label=['nrooms'], xaxis_label='nrooms_bins', labels=labels, visualize=visualize)

    title = 'Percentage of neurons active in n rooms'
    axis_labels = ['Number of rooms', 'Percentage of PCs']
    print("Plotting results...")
    namea = name + '-' + 'nrooms'
    make_plot('line', newdf, title, axis_labels, namea, ynormalized=False)

if plot == 'remapping' or plot == 'overlap':

    visualize = ['arg_encoding', 'arg_embedding_sigma']
    labels = ['e', 's']
    params_sweep = [('rotation', 1),('parallel',1),('flexible',1)]
    tags = ['overlap', 'overlapshuff']
    
    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=False, data_label = tags, xaxis_label= 'arg_encoding', labels= labels, visualize=visualize)
    
    title = 'FR overlap across environments'
    axis_labels = ['', 'Overlap'] 
    print("Plotting results...")
    namea = name + '-' + 'overlap'
    make_plot('bars_shuff', newdf, title, axis_labels, namea, ynormalized=True)

if plot == 'remapping' or plot == 'spatialcorr':

    visualize = ['arg_encoding', 'arg_embedding_sigma']
    labels = ['e', 's']
    params_sweep = [('rotation', 1),('parallel',1),('flexible',1)]
    tags = ['spatialcorr', 'spatialcorrshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(fdf, across=False, data_label = tags, xaxis_label= 'arg_encoding', labels= labels, visualize=visualize)
    
    title = 'Spatial correlation of place fields across environments'
    axis_labels = ['', 'Spatial correlation'] 
    print("Plotting results...")
    namea = name + '-' + 'spatialcorr'
    make_plot('bars_shuff', newdf, title, axis_labels, namea, ynormalized=True)

if plot == 'remapping' or plot == 'variance_remap':

    visualize = ['arg_encoding', 'arg_embedding_sigma']
    labels = ['e', 's']
    params_sweep = [(e,s) for e in {'rotation'} for s in [0,0.2,0.4,0.6000000000000001,0.8,1]]
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

if plot == 'remapping' or plot == 'multispatialcorr':

    visualize = ['arg_encoding', 'arg_embedding_sigma']
    labels = ['e', 's']
    params_sweep = [('flexible',1)]
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
    params_sweep = [(16,512,0, 'flexible')]
    tags = ['frdistance_pertuning', 'frdistance_pertuningshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Distance of FRs vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Distance of FRs'] 
    print("Plotting results...")
    namea = name + '-' + 'frdistance_pertuning'
    make_plot('scatter', newdf, title, axis_labels, namea)

if plot == 'remapping' or plot == 'spatialcorr_pertuning':

    visualize = ['arg_encoding', 'arg_embedding_sigma']
    labels = ['e', 's']
    params_sweep = [('flexible',1)]
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
    params_sweep = [(16,512,0,'flexible')]
    tags = ['nrooms_pertuning', 'nrooms_pertuningshuff']

    fdf = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(fdf, tags,['data','datashuff'], labels, visualize)
    
    title = 'Number of rooms vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Number of rooms'] 
    print("Plotting results...")
    namea = name + '-' + 'nrooms_pertuning'
    make_plot('scatter', newdf, title, axis_labels, namea)

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
