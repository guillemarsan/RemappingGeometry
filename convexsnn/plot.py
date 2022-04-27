import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D as plt3d

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

def plot_2dbboxproj(x, y, F, G, T, basepath, plotx = False):

    plt.figure(figsize=(10,10))
    n = F.shape[0]
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    y1 = y[0,:]
    y2 = y[1,:]
    lmin = np.min(y[:2,:])-3
    lmax = np.max(y[:2,:])+3
    lim = np.linspace(lmin, lmax, 500)
    lim1, lim2 = np.meshgrid(lim,lim)
    plane_func = lambda i, pxs, pys: (F[i,:] @ x - T[i] - G[i,0] * pxs - G[i,1]*pys)/G[i,2]

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

    plt.scatter(y1,y2, c=np.arange(y.shape[1]), cmap='jet')
    plt.scatter(y1[-1], y2[-1])
    if plotx:
        plt.scatter(x[0],x[1], marker='+', c='k')

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

    plt.figure(figsize=(10,10))
    t *= 1000
    ax1 = plt.subplot(2, 1, 1)
    for i in range(x.shape[0]):
        ax1.plot(t,x[i,:], label= 'x' + str(i) if i < 16 else '_nolegend_')
    for i in range(y.shape[0]):
        ax1.plot(t,y[i,:], label='y' + str(i) if i < 16 else '_nolegend_')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('x,y(mV)')
    ax1.set_xlabel('t(ms)')


    ax2 = plt.subplot(2, 1, 2)
    maxV = np.max(V)
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    l = 0.3
    off = 0.1
    for i in range(V.shape[0]):
        ax2.plot(t,V[i,:], label='V' + str(i) if i < 16 else '_nolegend_', color=c[i%10])
        ax2.vlines(t[s[i,:] == 1], maxV + (i+1)*off + i*l, maxV + (i+1)*off + (i+1)*l, color=c[i%10])
    ax2.legend(loc='upper right')
    ax2.set_ylabel('V(mV)')
    ax2.set_xlabel('t(ms)')

    filepath = "%s-neurosc.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    return plt.gcf()

def plot_1drfs(p, r, dt, basepath, pad=0):

    p = p[pad:]
    r = r[:,pad:]

    plt.figure(figsize=(10,10))
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tf = 0.2
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

    x = x[pad:]

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

    p = p[pad:]
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


    
def plot_2drfs(p, r, dt, basepath):
    n = r.shape[0]

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

    fig2 = plt.figure(figsize=(10,10))
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    tf = 0.2
    m = int(tf/dt)
    filter = np.ones(m)*1/m
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        rf = np.convolve(r[i,:],filter,'same')
        plt.scatter(p[0,:],p[1,:],c=rf, cmap='jet',s=1)
        plt.title('Neuron %i' % i)
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


def plot_2dspikebins(p, s, b, basepath, grid=True):

    plt.figure(figsize=(10,10))
    n = s.shape[0]
    radius = (np.max(p)-np.min(p))/2
    step = 2*radius/np.sqrt(b)

    if grid:
        rows = np.arange(-radius+step/2, radius, step=step)
        ptr = rows.shape[0]
        pts = np.arange(ptr**2)
        ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T
    else:
        binsize = int(s.shape[1]/b)
        pts = np.arange(b)
        ppts = p[:,pts*binsize]
    
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    for i in np.arange(n):
        plt.subplot(nrows, ncols, i+1)
        sums = np.zeros(pts.shape[0])
        for j in pts:
            if grid:
                dist = np.max(np.abs(np.expand_dims(ppts[:,j],-1)-p), axis=0)
            else:
                dist = np.linalg.norm(np.expand_dims(ppts[:,j],-1)-p, axis=0)
            consider = np.argwhere(dist < (step/2))
            sums[j] = np.sum(s[i,consider])
        if grid:
            plt.imshow(np.reshape(sums, (ptr,ptr)), cmap='jet', interpolation='bilinear')
        else:
            plt.scatter(ppts[0,:],ppts[1,:],c=sums, cmap='jet')
        plt.title('Neuron %i' % i)

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

def plot_errorplot(data, xaxis, title, labels, basepath):

    plt.figure(figsize=(10,10))
    for key, res in data.items():
        
        mean = np.mean(res, axis=1)
        err = np.std(res, axis=1)
        plt.errorbar(xaxis, mean, err, marker='.', capsize=3, label=key)

    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.ylim(0,1.1)
    plt.legend()

    plt.tight_layout()
    filepath = "%s-pcs_perc.png" % basepath
    plt.savefig(filepath, dpi=600, bbox_inches='tight')



