from operator import ne
import pathlib, argparse, json, time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time

from convexsnn.plot import plot_scatterplot, plot_violinplot, plot_displot, plot_errorplot_cond


############# PLOTTING ##########################################
def make_plot(ptype, data, title, axis_labels, basepath, ynormalized=False, equal=False):

    plt.figure(figsize=(4,4))
    ax = plt.gca()

    if ptype == 'line': 
        for _, point in data.iterrows():
            mean = point['mean']
            err = point['std']
            xaxis = point['xaxis']
            legend = point['legend']
            plt.errorbar(xaxis, mean, err, marker='.', capsize=3, label=legend)
    
    elif ptype == 'hist':
        for _, point in data.iterrows():
            hist = point['hist']
            bins = point['bins']
            legend = point['legend']

            widths = np.diff(bins)
            xticks = bins[:-1] + widths/2
            plt.bar(xticks, hist, width=widths, align='center', label=legend)

    elif ptype == 'bars':
        i = 1
        conditions = []
        for _, point in data.iterrows():
            data = point['data']
            mean = np.mean(data)
            std = np.std(data)
            datashuff = point['datashuff']
            meanshuff = np.mean(datashuff)
            conditions.append(point['legend'])

            plt.bar(i, mean, width=0.5, align='center')
            plt.errorbar(i, mean, std, ecolor='black',capsize=10)
            plt.hlines(meanshuff, i-0.6, i+0.6, colors='r')
            i += 1

        plt.xticks(np.arange(i-1)+1, conditions)

    elif ptype == 'scatter':
        for _, point in data.iterrows():
            data = point['data']
            datashuff = point['datashuff']

            
            plt.scatter(data[:,0],data[:,1], alpha=0.7, label='Pairs')
            plt.scatter(datashuff[:,0], datashuff[:,1], alpha=0.7, label='Shuffle')

        plt.xlim(0,1.1)

    elif ptype == 'pfs':

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
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

            maxFR = np.max(pfs)
            for cell in np.arange(num_neurons):

                cmap = colors.ListedColormap(['white', c[cell]])
                bounds=[0,0,maxFR]
                norm = colors.BoundaryNorm(bounds, cmap.N)

                pf = pfs[cell,:]
                sqrtb = int(np.sqrt(pfs.shape[1]))
                pf = pf.reshape((sqrtb,sqrtb))
                alphas = pf/np.max(pf) if np.max(pf) > 0 else 0
                im = axsub.imshow(pf, alpha=alphas, cmap=cmap, norm=norm)
            i += 1

        patches = [matplotlib.patches.Patch(color=c[i], label='N%i' % n) for i,n in enumerate(neurons) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. , prop={'size': 6})
    if ptype != 'pfs':
        plt.title(title, fontsize=10)
        plt.xlabel(axis_labels[0], fontsize=10)
        plt.ylabel(axis_labels[1], fontsize=10)
        if ynormalized: plt.ylim(0,1.1)
        if equal: plt.gca().set_aspect('equal')    
        plt.tick_params(axis='both', labelsize=10)
        plt.legend(frameon=False, prop={'size': 10})
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

    plt.tight_layout()
    filepath = "{0}-{1}_plot.png".format(basepath,ptype)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


############# LOADING AND PREPARING DATA ########################
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

def prepare_combine(df, across, data_label, xaxis_label):
    def combinelambda(set):
        points = pd.DataFrame([])
        for dlabel in data_label:
            newpoint = {}
            list = set[dlabel]
            list = list.apply(lambda x: np.array(eval(x)))

            if across:
                means = np.mean(list._values)
                stds = np.std(list._values)
                xaxis = np.array(eval(set.iloc[0][xaxis_label]))
            else:
                means = []
                stds = []
                xaxis = []
                i = 0
                for elem in list._values:
                    means.append(np.mean(elem))
                    stds.append(np.std(elem))
                    xaxis.append(set.iloc[i][xaxis_label])
                    i += 1
            newpoint['mean'] = [means]
            newpoint['std'] = [stds]
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
    parser.add_argument("--dir", type=str, default='TestPerTuning',
                        help="Directory to read and write files")
    parser.add_argument("--plot", type=str, default='spatialcorr_pertuning',
                        help = 'Which plot to make')
    

    args = parser.parse_args()

plot = args.plot

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = './data/' + args.dir + '/'
name = basepath +  timestr + '-' + args.dir + '-' + plot
    
print("Loading results...")

# Load DataFrame
if plot in {'placefields'}:
    df = load_dataframe('database', basepath)
elif plot in {'spatialinfo'}:
    df = load_dataframe('placecells', basepath)
elif plot in {'nrooms', 'overlap', 'spatialcorr', 'variance_remap', 'multispatialcorr', 'frdistance_pertuning', 'spatialcorr_pertuning'}:
    df = load_dataframe('remapping', basepath)
else:
    df = load_dataframe('recruitment', basepath)

# Plotting
print ("Plotting results...")

#### General plots ################

if plot == 'placefields':

    visualize = ['arg_env', 'arg_encoding']
    labels = ['e', 'enc']
    params_sweep = [(0,'parallel'), (3,'parallel')]
    tags = ['pfs']
    neurons = np.array([161,162,190,203])
    
    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_pfs(df, neurons, visualize, labels)

    title = ''
    axis_labels = []
    print("Plotting results...")
    make_plot('pfs', newdf, title, axis_labels, name)



#### Place cell plots #############
elif plot == 'spatialinfo':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id', 'arg_encoding']
    labels = ['d', 'e', 'n', 'l']
    params_sweep = [(8,0,256,0,'rotation')]
    tags = ['spatialinfo', 'spatialinfo_bins']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['hist','bins'], labels, visualize)
    

    title = 'Spatial information distribution'
    axis_labels = ['Spatial information (bits/s)', 'Percentage of neurons']
    print("Plotting results...")
    make_plot('hist', newdf, title, axis_labels, name, ynormalized=False)

#### Remapping plots #############
elif plot == 'nrooms':

    visualize = ['arg_dim_bbox','arg_nb_neurons']
    labels = ['d', 'n']
    params_sweep = [(8,256)]
    tags = ['nrooms', 'nrooms_bins']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_combine(df, across=True, data_label=['nrooms'], xaxis_label='nrooms_bins')

    title = 'Percentage of neurons active in n rooms'
    axis_labels = ['Number of rooms', 'Percentage of PCs']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'overlap':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_encoding']
    labels = ['d', 'n', 'l']
    params_sweep = [(16,512,'rotation'),(16,512,'parallel'),(16,512,'flexible')]
    tags = ['overlap', 'overlapshuff']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['data','datashuff'], labels, visualize)
    
    title = 'Mean FR correlation of place cells across environments'
    axis_labels = ['Condition', 'Mean FR correlation (cosine of angle)'] 
    print("Plotting results...")
    make_plot('bars', newdf, title, axis_labels, name, ynormalized=True)

elif plot == 'spatialcorr':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_encoding']
    labels = ['d', 'n', 'l']
    params_sweep = [(16,512,'rotation'),(16,512,'parallel'),(16,512,'flexible')]
    tags = ['spatialcorr', 'spatialcorrshuff']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['data','datashuff'], labels, visualize)
    
    title = 'Mean FR correlation of place cells across environments'
    axis_labels = ['Condition', 'Mean FR correlation (cosine of angle)'] 
    print("Plotting results...")
    make_plot('bars', newdf, title, axis_labels, name, ynormalized=True)

elif plot == 'variance_remap':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'l']
    params_sweep = [(8,256,0)]
    tags = ['overlap', 'overlapshuff', 'spatialcorr', 'spatialcorrshuff']

    df = filter(df, visualize, params_sweep, tags)

    # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'l'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)
    
    title = 'Overlap and spatial correlation per embedding variance'
    axis_labels = ['Variance', 'Metric'] 
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=True)

elif plot == 'multispatialcorr':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'l']
    params_sweep = [(16,512,0)]
    tags = ['multispatialcorr', 'multispatialcorrshuff']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['data','datashuff'], labels, visualize)
    
    title = 'Pair of cells spatial correlation in two environements'
    axis_labels = ['Environment 1', 'Environment 2'] 
    print("Plotting results...")
    make_plot('scatter', newdf, title, axis_labels, name, ynormalized=True, equal=True)

elif plot == 'frdistance_pertuning':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'l']
    params_sweep = [(16,512,0)]
    tags = ['frdistance_pertuning', 'frdistance_pertuningshuff']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['data','datashuff'], labels, visualize)
    
    title = 'Distance of FRs vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Distance of FRs'] 
    print("Plotting results...")
    make_plot('scatter', newdf, title, axis_labels, name)

elif plot == 'spatialcorr_pertuning':

    visualize = ['arg_dim_bbox','arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'l']
    params_sweep = [(16,512,0)]
    tags = ['spatialcorr_pertuning', 'spatialcorr_pertuningshuff']

    df = filter(df, visualize, params_sweep, tags)
    newdf = prepare_single(df, tags,['data','datashuff'], labels, visualize)
    
    title = 'Spatial correlation vs place tuning alignment'
    axis_labels = ['Place tuning alignment', 'Spatial corr.'] 
    print("Plotting results...")
    make_plot('scatter', newdf, title, axis_labels, name)

#### Recruitment plots #############

elif plot == 'meanfr_perclass':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['meanfr_permanent', 'meanfr_tagged', 'meanfr_recruited', 'meanfr_silent']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Mean FR per class'
    axis_labels = ['Inhibition current', 'Mean FR (Hz)',]
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'decoding_error':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['decoding_error']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Decoding error with Inhibition'
    axis_labels = ['Inhibition current', 'Decoding error']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'fr':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['meanfr', 'maxfr']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Global FR with Inhibition'
    axis_labels = ['Inhibition current', 'FR (Hz)']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'pfsize':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['pfsize']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Place field size with Inhibition'
    axis_labels = ['Inhibition current', 'Place field size (m2)']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'pfsize_perclass':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['pfsize_permanent', 'pfsize_tagged','pfsize_recruited','pfsize_silent']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Place field size per class with Inhibition'
    axis_labels = ['Inhibition current', 'Place field size (m2)']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'perpcs_perclass':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,0),(4,0,64,1)]
    tags = ['perpcs_permanent', 'perpcs_tagged','perpcs_recruited','perpcs_silent']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Proportion of PCs in each class with Inhibition'
    axis_labels = ['Inhibition current', 'Proportion']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=False)

elif plot == 'coveredarea_perclass':

    visualize = ['arg_dim_bbox', 'arg_env', 'arg_nb_neurons', 'arg_load_id']
    labels = ['d', 'n', 'I']
    params_sweep = [(4,0,64,1),(4,0,64,1)]
    tags = ['coveredarea', 'coveredarea_permanent', 'coveredarea_tagged','coveredarea_recruited','coveredarea_silent']

    df = filter(df, visualize, params_sweep, tags)

     # Xaxis variable
    xaxis = 'arg_load_id'
    xaxislabel = 'I'
    visualize.remove(xaxis)
    labels.remove(xaxislabel)

    newdf = prepare_combine(df, across=False, data_label=tags, xaxis_label=xaxis)

    title = 'Decoding error with Inhibition'
    axis_labels = ['Inhibition current', 'Place field size (m2)']
    print("Plotting results...")
    make_plot('line', newdf, title, axis_labels, name, ynormalized=True)