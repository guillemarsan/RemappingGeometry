import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from utils import compute_meshgrid, compute_ratemap, compute_pf
from mpl_toolkits.mplot3d import Axes3D as plt3d
from scipy.stats.stats import pearsonr

def plot_iofunc(x, y, F, G, T, basepath):
    
    plt.figure(figsize=(10,10))
    n = F.shape[0]
    lim = np.linspace(np.min(x)-1, np.max(x)+1, 50)
    for i in range(n):
        line = (F[i] * lim - T[i])/ G[i]
        plt.plot(lim, line)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(lim,np.zeros_like(lim))
    plt.scatter(x,y,c=np.arange(y.shape[1]), cmap='jet')
    plt.scatter(x[:,-1], y[:,-1])

    filepath = "%s-iofunc.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()


def plot_1dbbox(x, y, F, G, T, basepath, plotx = False, foranim=False):

    if not foranim:
        plt.figure(figsize=(10,10))
    n = F.shape[0]
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y1 = y[0,:]
    y2 = y[1,:]
    lmin = np.min(y)-1
    lmax = np.max(y)+1
    lim = np.linspace(lmin, lmax, 100)
    line_func = lambda i, pxs: (F[i,:] @ x - T[i] - G[i,0] * pxs)/G[i,1]
    for i in range(n):
        line = line_func(i,lim)
        if foranim and y2 == line_func(i,y1):
            plt.plot(lim, line, color=c[i%10], linewidth = 10)
        else:
            plt.plot(lim, line, color=c[i%10])
        if not foranim:
            px,py = np.meshgrid(lim,lim)
            plt.quiver(0, line_func(i,0), G[i,0],G[i,1],width=0.005,headwidth=5)
            plt.imshow((np.sign(G[i,1])*py > np.sign(G[i,1])*line_func(i,px)).astype(int), extent=(px.min(),px.max(),py.min(),py.max()),
                cmap=colors.ListedColormap([c[i%10], 'white']), aspect='auto', origin="lower", alpha=0.3)
        
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.xlim(lmin,lmax)
    plt.ylim(lmin,lmax)
    plt.scatter(y1,y2, c=np.arange(y.shape[1]), cmap='jet')
    plt.scatter(y1[-1], y2[-1])
    if plotx:
        plt.scatter(x[0],x[1], marker='+', c='k')
    plt.gca().set_aspect('equal', adjustable='box')

    if not foranim:
        filepath = "%s-1dbbox.png" % basepath
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_2dbboxproj(F, G, T, A, basepath):

    plt.figure(figsize=(10,10))
    n = F.shape[0]
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']

    lmin = -A
    lmax = A
    lim = np.linspace(lmin, lmax, 500)
    lim1, lim2 = np.meshgrid(lim,lim)
    plane_func = lambda i, pxs, pys: (F[i,:] @ np.zeros((F.shape[1])) - T[i] - G[i,0] * pxs - G[i,1]*pys)/G[i,2]

    planes = np.zeros((500,500,n))
    for i in range(n):
        planes[:,:,i] = plane_func(i,lim1, lim2)

    maxidx = np.argmax(planes,axis=2)
    for i in range(n):
        plt.imshow((maxidx == i).astype(int), extent=(lmin,lmax,lmin,lmax),
                cmap=colors.ListedColormap([c[i%10]]), aspect='auto', origin="lower", alpha=(maxidx==i).astype(float)*0.75)
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.xlim(lmin,lmax)
    plt.ylim(lmin,lmax)

    filepath = "%s-2dbboxproj.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()


def plot_2dbbox(x, y, F, G, T, basepath, plotx = False, foranim=False):

    if not foranim:
        plt.figure(figsize=(10,10))
        ax3 = plt.axes(projection='3d')
    n = F.shape[0]
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y1 = y[0,:]
    y2 = y[1,:]
    y3 = y[2,:]
    lmin = np.min(y)-1
    lmax = np.max(y)+1
    lim = np.linspace(lmin, lmax, 100)
    lim1, lim2 = np.meshgrid(lim,lim)
    plane_func = lambda i, pxs, pys: (F[i,:] @ x - T[i] - G[i,0] * pxs - G[i,1]*pys)/G[i,2]

    planes = np.zeros((100,100,n))
    for i in range(n):
        planes[:,:,i] = plane_func(i,lim1, lim2)

    maxidx = np.argmax(planes,axis=2)
    for i in range(n):
        mins = np.argwhere(maxidx != i)
        planes[mins[:,0],mins[:,1],i] = np.nan

        if foranim and y3 == plane_func(i,y1,y2):
            ax3.plot_surface(X=lim1, Y=lim2, Z=planes[:,:,i], color=c[i%10])
        else:
            ax3.plot_surface(X=lim1, Y=lim2, Z=planes[:,:,i], color=c[i%10], alpha=0.3)

    plt.xlabel('y1')
    plt.ylabel('y2')
    ax3.set_zlabel('y3')
    plt.xlim(lmin,lmax)
    plt.ylim(lmin,lmax)
    ax3.set_zlim(lmin,lmax)
    ax3.scatter(y1,y2,y3, c=np.arange(y.shape[1]), cmap='jet')
    ax3.scatter(y1[-1], y2[-1], y3[-1])
    if plotx:
        ax3.scatter(x[0],x[1], x[2], marker='+', c='k')

    if not foranim:
        filepath = "%s-2dbbox.png" % basepath
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1danimbbox(x, y, F, G, T, basepath, plotx = False):

    fig, ax = plt.subplots(figsize=(10,10))
    fr = 1000
    step = int(x.shape[1]/fr)
    minx = np.min(x[0,:]) - 0.5
    maxx = np.max(x[0,:]) + 0.5
    miny = np.min(x[1,:]) - 0.5
    maxy = np.max(x[1,:]) + 0.5

    def animate(i):
        ax.clear()
        f = i*step
        plot_1dbbox(x[:,f], np.expand_dims(y[:,f],-1), F, G, T, basepath, plotx, foranim=True)
        plt.xlim(minx, maxx)
        plt.ylim(miny, maxy)
        return ax 
        
    ani = animation.FuncAnimation(fig, animate, frames=fr)

    filepath = "%s-bboxmov.gif" % basepath        
    ani.save(filepath, fps=int(fr/10))         
    return ani

def plot_neuroscience(x, y, V, s, t, basepath):

    plt.figure(figsize=(20,10))
    t_ms = t * 1000
    ax1 = plt.subplot(3, 1, 1)
    for i in range(x.shape[0]):
        ax1.plot(t_ms,x[i,:], label= 'x' + str(i) if i < 16 else '_nolegend_')
    for i in range(y.shape[0]):
        ax1.plot(t_ms,y[i,:], label='y' + str(i) if i < 16 else '_nolegend_')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('x,y(mV)')
    ax1.set_xlabel('t(ms)')


    ax2 = plt.subplot(3, 1, 2)
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    l = 0.3
    off = 0.1
    for i in range(V.shape[0]):
        ax2.plot(t_ms,V[i,:], label='V' + str(i) if i < 16 else '_nolegend_', color=c[i%10])
    ax2.legend(loc='upper right')
    ax2.set_ylabel('V(mV)')
    ax2.set_xlabel('t(ms)')

    ax3 = plt.subplot(3, 1, 3)
    for i in range(V.shape[0]):
        ax3.vlines(t_ms[s[i,:] == 1], (i+1)*off + i*l, (i+1)*off + (i+1)*l, color=c[i%10])
    ax3.set_ylabel('Neuron')
    ax3.set_xlabel('t(ms)')

    filepath = "%s-neurosc.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_pathtrajectory(p, s, t, basepath):

    plt.figure(figsize=(10,10))
    if p.shape[0] == 1:
        plt.plot(t,p[0,:],color='black')
        for i in range(s.shape[0]):
            t_spikes = np.where(s[i,:])
            plt.scatter(t[t_spikes],p[0,t_spikes])
        plt.ylabel('p(m)')
        plt.xlabel('t(ms)')
    else:
        plt.plot(p[0,:],p[1,:],color='black')
        for i in range(s.shape[0]):
            t_spikes = np.where(s[i,:])
            plt.scatter(p[0,t_spikes],p[1,t_spikes])
        plt.ylabel('p_1(m)')
        plt.xlabel('p_0(m)')
    

    filepath = "%s-pathtraj.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


    plt.figure(figsize=(10,10))
    n = s.shape[0]
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1, aspect='equal')

        plt.title('Neuron %i' % i)
        if p.shape[0] == 1:
            plt.plot(t,p[0,:],color='black')
            t_spikes = np.where(s[i,:])
            plt.scatter(t[t_spikes],p[0,t_spikes], color=c[i%10])
            plt.ylabel('p(m)')
            plt.xlabel('t(ms)')
        else:
            plt.plot(p[0,:],p[1,:],color='black')
            t_spikes = np.where(s[i,:])
            plt.scatter(p[0,t_spikes],p[1,t_spikes], color=c[i%10])
            plt.ylabel('p_1(m)')
            plt.xlabel('p_0(m)')
        

    plt.tight_layout()
    filepath = "%s-pathtraj_sep.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1drfs(p, r, dt, basepath, pad=0):

    p = p[0,pad:]
    r = r[:,pad:]

    plt.figure(figsize=(10,10))
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tf = 0.1
    m = int(tf/dt)
    filter = np.ones(m)*1/m
    for i in range(r.shape[0]):
        rf = np.convolve(r[i,:],filter, 'same')
        plt.plot(p, rf, label='r' + str(i) if i < 16 else '_nolegend_', color=c[i%10])
    plt.legend()
    plt.xlim(np.min(p),np.max(p))
    plt.ylabel('r')
    plt.xlabel('p')

    filepath = "%s-1drfs.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1drfsth(D, x, p, basepath, pad=0):

    p = p[0,pad:]
    x = x[:,pad:]

    plt.figure(figsize=(10,10))
    c = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    dotprod = D.T @ x
    for i in range(dotprod.shape[0]):
        plt.plot(p, dotprod[i,:], label='r' + str(i) if i < 16 else '_nolegend_', color=c[i%10])

    colors = c[np.argmax(dotprod, axis=0)%10]
    plt.scatter(p, np.ones_like(p)*np.max(dotprod)+0.05, color=colors)
    plt.legend()
    plt.xlim(np.min(p),np.max(p))
    plt.ylabel('r')
    plt.xlabel('p')

    filepath = "%s-1drfsth.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1dspikebins(p, s, b, basepath, pad=0):

    p = p[0,pad:]
    s = s[:,pad:]

    plt.figure(figsize=(10,10))
    n = s.shape[0]

    diam = np.max(p)-np.min(p)
    binsize = int(s.shape[1]/b)
    pts = np.arange(b)
    ppts = np.expand_dims(p[pts*binsize], -1)

    auxppts = np.tile(ppts,(1,p.shape[0]))
    auxp = np.tile(p,(ppts.shape[0],1))
    
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        dist = np.abs(auxppts-auxp)
        smat = np.tile(s[i,:],(b,1))
        sums = np.sum(smat * (dist < diam/b), axis=1)
        plt.bar(ppts[:,0], sums)
        plt.title('Neuron %i' % i)

    plt.tight_layout()
    filepath = "%s-1dspikebins.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


    
def plot_2drfs(p, r, dt, basepath, n_vect):

    fig1 = plt.figure(figsize=(10,10))
    c = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    maxs = np.max(r, axis=0)
    colors = c[np.argmax(r, axis=0)%10]
    colors[maxs == 0] = 'k'
    plt.scatter(p[0,:], p[1,:], color=colors)
    plt.ylabel('p2')
    plt.xlabel('p1')

    filepath = "%s-2drfsmax.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    
    n = n_vect.shape[0]

    fig2 = plt.figure(figsize=(10,10))
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    tf = 0.01
    m = int(tf/dt)
    filter = np.ones(m)*1/m
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        rf = np.convolve(r[n_vect[i],:],filter,'same')
        plt.scatter(p[0,:],p[1,:],c=rf, cmap='jet',s=1)
        plt.title('Neuron %i' % n_vect[i])
        plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    filepath = "%s-2drfs.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

    return fig1, fig2

def plot_2drfsth(D, x, p, basepath):

    dotprod = D.T @ x

    fig1 = plt.figure(figsize=(10,10))
    c = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    maxs = np.max(dotprod, axis=0)
    colors = c[np.argmax(dotprod, axis=0)%10]
    colors[maxs == 0] = 'k'
    plt.scatter(p[0,:], p[1,:], color=colors)
    plt.ylabel('p2')
    plt.xlabel('p1')

    filepath = "%s-2drfsthmax.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

    return fig1


def plot_2dspikebins(p, s, dt, b, basepath, n_vect, maxfr=None, grid=True):

    plt.figure(figsize=(7,7))
    n = n_vect.shape[0]

    radius = (np.max(p)-np.min(p))/2
    step = 2*radius/np.sqrt(b)

    if grid:
        bins = compute_meshgrid(radius, b)
    else:
        binsize = int(s.shape[1]/b)
        pts = np.arange(b)
        ppts = p[:,pts*binsize]
    
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        sums = np.zeros(bins.shape[1])
        if grid:
            ratemap = compute_ratemap(p, s[n_vect[i]], dt, bins)
            sums = compute_pf(ratemap, bins)
        else:
            for j in np.arange(bins.shape[1]):
                dist = np.linalg.norm(np.expand_dims(ppts[:,j],-1)-p, axis=0)
                consider = np.argwhere(dist < (step/2))
                sums[j] = np.sum(s[n_vect[i],consider])/(consider.size*dt) if consider.size != 0 else 0
        if grid:
            ptr = int(np.sqrt(b))
            plt.imshow(np.reshape(sums, (ptr,ptr)), cmap='jet', interpolation='bilinear')
        else:
            plt.scatter(ppts[0,:],ppts[1,:],c=sums, cmap='jet')

        plt.xticks([])
        plt.yticks([])
        if maxfr is not None:
            plt.title('N %i' % n_vect[i] + ',%.2fHz' % maxfr[i], fontsize=10)
        else:
            plt.title('N %i' % n_vect[i], fontsize=10)

    plt.tight_layout()
    filepath = "%s-2dspikebins.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


def plot_2dspikebinstmp(p, s, b, dt, basepath):

    plt.figure(figsize=(10,10))
    n = s.shape[0]
    timesteps = s.shape[1]

    binsize = int(timesteps/b)
    bintime = binsize*dt
    bin_vect = np.arange(b)*binsize
    bin_vect = np.append(bin_vect,timesteps-1)

    pbin = (bin_vect[:-1] + bin_vect[1:])/2
    meanp = p[:,pbin.astype(int)]
    
    nrows = int(np.ceil(n/3))
    for i in np.arange(n):
        plt.subplot(nrows, 3, i+1)
        hist, _ = np.histogram(np.where(s[i,:] > 0), bins=bin_vect)
        frs = hist/bintime
        plt.scatter(meanp[0,:],meanp[1,:],c=frs, cmap='jet')
        plt.title('Neuron %i' % i)

    plt.tight_layout()
    filepath = "%s-2dspikebins.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

def plot_errorplot(data, xaxis, title, labels, basepath, ground=None, ynormalized=True):

    plt.figure(figsize=(4,4))
    ax = plt.gca()
    for key, res in data.items():
        
        mean = np.mean(res, axis=1)
        err = np.std(res, axis=1)
        plt.errorbar(xaxis, mean, err, marker='.', capsize=3, label=key)

    if ground is not None:
        plt.plot(ground[0,:], ground[1,:], label='analytical')

    plt.title(title, fontsize=10)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(labels[1], fontsize=10)
    if ynormalized: plt.ylim(0,1.1)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(frameon=False, prop={'size': 10})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filepath = "%s-error_plot.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

def plot_scatterplot(data, title, labels, basepath, separated_fits=True):

    color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_idx = 0

    total_px = np.array([])
    total_py = np.array([])
    plt.figure(figsize=(5,5))
    for key, res in data.items():
        px_vect = res[0,:]
        py_vect = res[1,:]

        plt.scatter(px_vect, py_vect, color=color_list[color_idx % 10], label=key)
        if separated_fits:
            coeffs = np.polyfit(px_vect,py_vect,1)
            samples = np.linspace(np.min(px_vect)-0.01,np.max(px_vect)+0.01,100)
            pyhat = coeffs[1] + coeffs[0]*samples
            plt.plot(samples,pyhat,color=color_list[color_idx % 10])
        else:
            total_px = np.append(total_px,px_vect)
            total_py = np.append(total_py,py_vect)
        color_idx += 1
    
    if not separated_fits:
        coeffs = np.polyfit(total_px,total_py,1)
        samples = np.linspace(np.min(px_vect)-0.01,np.max(px_vect)+0.01,100)
        pyhat = coeffs[1] + coeffs[0]*samples
        plt.plot(samples,pyhat,color=color_list[color_idx % 10])
    

    plt.title(title, fontsize=10)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(labels[1], fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylim(bottom=0)
    plt.legend()

    plt.tight_layout()
    filepath = "%s-scatter.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

def plot_step(data, title, labels, basepath):

    color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_idx = 0
    nbins_pcs = np.arange(0,1,0.02)
    nbins_embedds = np.arange(-1,1,0.02)

    plt.figure(figsize=(5,5))
    for key, res in data.items():
        px_vect = res[0,:]
        py_vect = res[1,:]

        
        hist_pcs = np.histogram(py_vect, bins=nbins_pcs)[0]/py_vect.shape[0]
        hist_embedds = np.histogram(px_vect, bins=nbins_embedds)[0]/px_vect.shape[0]
        plt.figure("f2")
        plt.step(nbins_pcs[:-1], hist_pcs, color=color_list[color_idx],where="post",linewidth=1,label=key)
        plt.figure("f3")
        plt.step(nbins_embedds[:-1], hist_embedds, color=color_list[color_idx],where="post",linewidth=1,label=key)

        color_idx += 1

    plt.title(title, fontsize=10)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(labels[1], fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylim([0,1])
    plt.xlim(left=0)
    plt.legend()

    plt.tight_layout()
    filepath = "%s-freq_diffs.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

def plot_violinplot(data, title, labels, basepath):

    color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    color_idx = 1

    total_px = np.array([])
    total_py = np.array([])
    plt.figure(figsize=(5,5))
    for key, res in data.items():
        px_vect = res[0,:]
        py_vect = res[1,:]
        total_px = np.append(total_px,px_vect)
        total_py = np.append(total_py,py_vect)
    
    xticks = np.unique(total_px)
    dataset = []
    means =[]
    stds = []
    for xt in xticks:
        pointsxt = total_py[total_px == xt]
        dataset.append(pointsxt)
        means.append(np.mean(pointsxt))
        stds.append(np.std(pointsxt))


    plt.violinplot(dataset, showextrema=False)
    plt.errorbar(xticks, means, stds, linestyle='None', fmt='o', capsize=5, color='k')

    coeffs = np.polyfit(total_px,total_py,1)
    Rp = pearsonr(total_px, total_py)
    plt.text(0.6, 0.9, "R = {:.4f} \n p = {:.4f}".format(Rp[0], Rp[1]), transform=plt.gca().transAxes)
    samples = np.linspace(np.min(px_vect)-0.01,np.max(px_vect)+0.01,100)
    pyhat = coeffs[1] + coeffs[0]*samples
    plt.plot(samples,pyhat,color=color_list[color_idx % 10])
    

    plt.title(title, fontsize=10)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(labels[1], fontsize=10)
    plt.tick_params(axis='both', labelsize=10)
    plt.ylim(bottom=0)
    plt.legend()

    plt.tight_layout()
    filepath = "%s-scatter.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')

def plot_displot(data, xaxis, title, labels, basepath, ground=None, ynormalized=True):
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    for key, res in data.items():
        
        mean = np.mean(res, axis=1)
        err = np.std(res, axis=1)
        
        if len(xaxis)-mean.shape[0] > 0:
            zeros = np.zeros(len(xaxis)-mean.shape[0])
            mean = np.concatenate((mean, zeros))

        plt.bar(xaxis, mean, width=np.abs(xaxis[1]-xaxis[0]), align='center')

    if ground is not None:
        plt.plot(ground[0,:], ground[1,:], label='analytical')

    plt.title(title, fontsize=10)
    plt.xlabel(labels[0], fontsize=10)
    plt.ylabel(labels[1], fontsize=10)
    if ynormalized: plt.ylim(0,1.1)
    plt.tick_params(axis='both', labelsize=10)
    plt.legend(frameon=False, prop={'size': 10})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filepath = "%s-dis_plot.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
