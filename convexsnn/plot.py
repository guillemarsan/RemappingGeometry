import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from utils import compute_meshgrid, compute_pathloc, compute_ratemap, compute_pf
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
        plt.xlim(minx-0.3, maxx)
        plt.ylim(miny-0.3, maxy)
        return ax 
        
    ani = animation.FuncAnimation(fig, animate, frames=fr)

    filepath = "%s-bboxmov.gif" % basepath        
    ani.save(filepath, fps=int(fr/10))         
    return ani

def plot_neuroscience(x, y, V, s, t, basepath, n_vect, T=None):

    plt.figure(figsize=(20,10))
    matplotlib.rcParams.update({'font.size': 22})
    t_ms = t * 1000
    ax1 = plt.subplot(3, 1, 1)
    for i in range(y.shape[0]):
        ax1.plot(t_ms,y[i,:], label='y' + str(i) if i < 16 else '_nolegend_')
    for i in range(x.shape[0]):
        ax1.plot(t_ms,x[i,:], label= 'x' + str(i) if i < 16 else '_nolegend_')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Input x')
    ax1.set_xlabel('Time(s)')


    ax2 = plt.subplot(3, 1, 2)
    #c = plt.rcParams['axes.prop_cycle'].by_key()['color'] # color=c[i%10]
    jet = cm = plt.get_cmap('hsv') 
    cNorm  = colors.Normalize(vmin=0, vmax=n_vect.shape[0])
    c = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    l = 0.3
    off = 0.1
    for i in range(n_vect.shape[0]):
        ax2.plot(t_ms,V[n_vect[i],:], label='V' + str(n_vect[i]) if i < 16 else '_nolegend_', color=c.to_rgba(i))
    if T is not None:
        ax2.plot(t_ms,T*np.ones_like(t_ms),'k--')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('V(mV)')
    ax2.set_xlabel('t(ms)')

    ax3 = plt.subplot(3, 1, 3)
    for i in range(n_vect.shape[0]):
        ax3.vlines(t_ms[s[n_vect[i],:] == 1], (i+1)*off + i*l, (i+1)*off + (i+1)*l, color=c.to_rgba(i))
    ax3.set_ylabel('Trial 1')
    ax3.set_xlabel('t(ms)')

    filepath = "%s-neurosc.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_pathtrajectory(p, s, t, basepath, n_vect, unique=None, phat=None):

    matplotlib.rcParams.update({'font.size': 22})
    if p.shape[0] == 1:
        plt.figure(figsize=(20,3))
        plt.plot(t,p[0,:],color='black',zorder=10, label='True position')
        if phat is not None:
            plt.plot(t,phat[0,:],color='b',alpha=0.5,zorder=0,label='Estimated position')
        if unique is None:
            for i in range(s.shape[0]):
                t_spikes = np.where(s[i,:])
                plt.scatter(t[t_spikes],p[0,t_spikes],zorder=20,s=4)
        else:
            t_spikes = np.where(s[unique,:])
            plt.scatter(t[t_spikes],p[0,t_spikes],zorder=20,s=6,c='r', label='Example neuron spikes')
        plt.ylabel('Position(m)')
        plt.xlabel('Time(s)')
        plt.legend()
        plt.ylim(-1,1)
    else:
        plt.figure(figsize=(10,10))
        if phat is not None:
            plt.plot(phat[0,:],phat[1,:],color='b',alpha=0.5,zorder=0)
        plt.plot(p[0,:],p[1,:],color='black',zorder=10)
        if unique is None:
            for i in range(s.shape[0]):
                t_spikes = np.where(s[i,:])
                plt.scatter(p[0,t_spikes],p[1,t_spikes],zorder=20,s=4)
        else:
            t_spikes = np.where(s[unique,:])
            plt.scatter(p[0,t_spikes],p[1,t_spikes],zorder=20,s=50,c='r')
        plt.ylabel('p_1(m)')
        plt.xlabel('p_0(m)')
        plt.xlim(-1,1)
        plt.ylim(-1,1)

    filepath = "%s-pathtraj.svg" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


    plt.figure(figsize=(10,10))
    n = n_vect.shape[0]
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in np.arange(n):
        if p.shape[0] == 1:
            plt.subplot(nrows, ncols, i+1)
            plt.plot(t,p[0,:],color='black',linewidth=0.5,zorder=0)
            t_spikes = np.where(s[n_vect[i],:])
            plt.scatter(t[t_spikes],p[0,t_spikes], color=c[i%10],zorder=10,s=0.75)
            plt.ylabel('p(m)')
            plt.xlabel('t(ms)')
            plt.ylim(-1,1)  
        else:
            plt.subplot(nrows, ncols, i+1, aspect='equal',zorder=0)
            plt.plot(p[0,:],p[1,:],linewidth=0.5,color='black')
            t_spikes = np.where(s[n_vect[i],:])
            plt.scatter(p[0,t_spikes],p[1,t_spikes], color=c[i%10],zorder=10,s=0.75)
            plt.ylabel('p_1(m)')
            plt.xlabel('p_0(m)')
            plt.xlim(-1,1)
            plt.ylim(-1,1)
        plt.title('Neuron %i' % n_vect[i])

    plt.tight_layout()
    filepath = "%s-pathtraj_sep.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_animtraj(p, s, t, basepath, neuron=-1):

    fig, ax = plt.subplots(figsize=(10,10))

    fr = int(t.shape[0]/1000)
    def animate(f_i):
        ax.clear()
        f = f_i*1000
        plt.plot(p[0,:f],p[1,:f],color='black',zorder=0)
        if neuron == -1:
            for i in range(s.shape[0]):
                t_spikes = np.where(s[i,:f])
                plt.scatter(p[0,t_spikes],p[1,t_spikes],zorder=10,s=4)
        else:
            t_spikes = np.where(s[neuron,:f])
            plt.scatter(p[0,t_spikes],p[1,t_spikes],zorder=10,s=6,c='r')
        plt.scatter(p[0,f],p[1,f])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        return ax 
        
    ani = animation.FuncAnimation(fig, animate, frames=fr)

    filepath = "%s-trajmov.gif" % basepath        
    ani.save(filepath, fps=10)         
    return ani


def plot_1drfs(p, fr, dt, basepath, n_vect):

    matplotlib.rcParams.update({'font.size': 22})
    plt.figure(figsize=(20,3))
    # c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    jet = cm = plt.get_cmap('hsv') 
    cNorm  = colors.Normalize(vmin=0, vmax=n_vect.shape[0])
    c = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    for i in range(n_vect.shape[0]):
        plt.plot(p[0,:], fr[n_vect[i],:], label='r' + str(n_vect[i]) if i < 16 else '_nolegend_', color=c.to_rgba(i))
    plt.legend()
    #plt.xlim(np.min(p),np.max(p))
    plt.ylabel('Firing rate (Hz)')
    plt.xlabel('Position (m)')
    plt.xticks

    filepath = "%s-1drfs.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1drfsth(D, x, p, basepath):

    plt.figure(figsize=(10,10))
    c = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    dotprod = D.T @ x
    for i in range(dotprod.shape[0]):
        plt.plot(p[0,:], dotprod[i,:], label='r' + str(i) if i < 16 else '_nolegend_', color=c[i%10])

    colors = c[np.argmax(dotprod, axis=0)%10]
    plt.scatter(p, np.ones_like(p)*np.max(dotprod)+0.05, color=colors)
    plt.legend()
    plt.xlim(np.min(p),np.max(p))
    plt.ylabel('r')
    plt.xlabel('p')

    filepath = "%s-1drfsth.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1dspikebins(p, s, b, basepath, n_vect):

    plt.figure(figsize=(10,10))
    n = s.shape[0]

    diam = np.max(p)-np.min(p)
    binsize = int(s.shape[1]/b)
    pts = np.arange(b)
    ppts = np.expand_dims(p[0,pts*binsize], -1)

    auxppts = np.tile(ppts,(1,p.shape[0]))
    auxp = np.tile(p,(ppts.shape[0],1))
    
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    for i in np.arange(n_vect.shape[0]):
        plt.subplot(nrows, ncols, i+1)
        dist = np.abs(auxppts-auxp)
        smat = np.tile(s[n_vect[i],:],(b,1))
        sums = np.sum(smat * (dist < diam/b), axis=1)
        plt.bar(ppts[:,0], sums)
        plt.xticks([])
        plt.yticks([])
        plt.title('Neuron %i' % n_vect[i], fontsize=10)

    plt.tight_layout()
    filepath = "%s-1dspikebins.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')


    
def plot_2drfs(p, fr, basepath, n_vect):

    fig1 = plt.figure(figsize=(10,10))
    c = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    maxs = np.max(fr, axis=0)
    colors = c[np.argmax(fr, axis=0)%10]
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
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        plt.scatter(p[0,:],p[1,:],c=fr[i,:], cmap='jet',s=1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Neuron %i' % n_vect[i], fontsize=10)
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


def plot_2dspikebins(p, s, dt, b, basepath, n_vect):

    plt.figure(figsize=(7,7))
    n = n_vect.shape[0]

    radius = (np.max(p)-np.min(p))/2
        
    bins = compute_meshgrid(radius, b)
    
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        sums = np.zeros(bins.shape[1])
       
        pathloc = compute_pathloc(p, bins)
        tb = np.sum(pathloc, axis=1)*dt
        ratemap = compute_ratemap(s[n_vect[i]], pathloc, tb)
        sums = compute_pf(ratemap, bins)
        maxfr = np.max(sums)
    
        ptr = int(np.sqrt(b))
        plt.imshow(np.reshape(sums, (ptr,ptr)), cmap='jet', interpolation='bilinear')
        plt.xticks([])
        plt.yticks([])
        plt.title('N %i' % n_vect[i] + ',%.2fHz' % maxfr, fontsize=10)

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

def plot_pe(p, eof, e, t, basepath):

    fig = plt.figure(figsize=(20,10))
    ax1 = plt.subplot(1, 2, 1)
    dim_pcs = p.shape[0]
    if dim_pcs == 1:
        ax1.plot(np.linspace(-1,1,eof.shape[0]), eof)
        ax1.set_ylim([-1,1])
        plt.title('e%i(p)' % e.shape[0])
    else:
        rows = int(np.sqrt(eof.shape[0]))
        step = (2/(rows-1))/2
        eofsquare = eof.reshape((rows,rows))
        im = ax1.imshow(np.flipud(eofsquare), extent=[-1-step, 1+step, -1-step, 1+step], vmin=-0.9, vmax=0.9)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.plot(p[0,:],p[1,:], 'k')
        plt.title('e%i(p)' % e.shape[0])
        

    ax2 = plt.subplot(1,2,2)
    for i in np.arange(e.shape[0]):
        ax2.plot(t,e[i,:], label='e' + str(i))
    ax2.set_ylim([-1,1])
    ax2.legend()

    filepath = "%s-gamma_pe.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_errorplot(data, xaxis, title, labels, basepath, ground=None, ynormalized=True, per_cond=None):

    plt.figure(figsize=(4,4))
    ax = plt.gca()
    i = 0
    for key, res in data.items():
        
        mean = res[0]
        err = res[1]
        if per_cond is None:
            plt.errorbar(xaxis, mean, err, marker='.', capsize=3, label=key)
        else:
            plt.errorbar(xaxis[i], mean, err, marker='.', capsize=3)
            i += 1

    if ground is not None:
        plt.plot(ground[0,:], ground[1,:], label='analytical')

    if per_cond is not None:
        plt.xticks(xaxis, labels=per_cond)
        plt.xlim(-1,xaxis.shape[0]+1)

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

def plot_errorplot_cond(data, tags, title, labels, basepath, ground='', ynormalized=True):

    plt.figure(figsize=(4,4))
    ax = plt.gca()
    i = 0
    for tag in tags:
        for key, res in data.items():
            if key.startswith(tag):
                mean = np.mean(res, axis=1)
                err = np.std(res, axis=1)
                if tag != ground:
                    plt.errorbar(i, mean, err, marker='.', capsize=3)
                    i += 1
                else:
                    tags.pop()
                    plt.errorbar(-0.5, mean, err, marker='.', color='r')
                    plt.hlines(mean, -1,len(tags)+1, color='r')

    plt.xticks(np.arange(len(tags)), labels=tags)
    plt.xlim(-1,len(tags)+1)

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

    # TODO move this to line fit elsewhere
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
        
        mean = res[0]
        err = res[1]
        
        if len(xaxis)-mean.shape[0] > 0:
            zeros = np.zeros(len(xaxis)-mean.shape[0])
            mean = np.concatenate((mean, zeros))

        if xaxis[0] == 0:
            plt.bar(xaxis[1:], mean[1:], width=np.abs(xaxis[3]-xaxis[2]), align='center')
            plt.bar(xaxis[0], mean[0], width=1, align='center')
        else:
            plt.bar(xaxis, mean, width=np.abs(xaxis[2]-xaxis[1]), align='center')

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
