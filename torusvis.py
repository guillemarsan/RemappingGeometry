import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import ortho_group
from convexsnn.AngleEncoder import AngleEncoder
from matplotlib import colors

timestr = time.strftime("%Y%m%d-%H%M%S")
basepath = './out/' + timestr
color_list = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def gnomonic_proj(x,y,z):

    theta = np.arctan2(y,x)
    phi = np.arctan2(np.sqrt(x**2 + y**2),z)
    r = np.tan(phi)
    xgno = r*np.cos(theta)
    ygno = r*np.sin(theta)

    return xgno, ygno

encoding = 'torus'
gnomonic = False
Dtype = 'load'
model = 'scn'
out = 2 if encoding == 'semicircle' else (3 if encoding in {'sphere', 'cylinder'} else 4)

if Dtype == 'load':
    n = 45
    filepath = './saved_bbox/seed' + str(0) + '/closed-load-polyae-dim-' + str(out) + '-n-' + str(n) + '-s-' + str(0) + '.npy'
    D = np.load(filepath)
elif Dtype == 'random':
    n = 20
    np.random.seed(1)
    Q = np.random.uniform(-1,1,size=(out,n))
    D = Q/np.linalg.norm(Q,axis=0)
elif Dtype == 'set':
    if encoding == 'semicircle':
        n = 5
        D = np.zeros((out,n))
        D[:,0] = [1,0]
        D[:,1] = [1,1]
        D[:,2] = [0,1]
        D[:,3] = [-1,1]
        D[:,4] = [-1,0]

    elif encoding in {'sphere', 'cylinder'}:
        n = 6
        D = np.zeros((out,n))
        D[:,0] = [0,-1,0]
        D[:,1] = [1,0,0]
        D[:,2] = [1,1,1]
        D[:,3] = [-1,1,1]
        D[:,4] = [0,-1,1]
        D[:,5] = [-0.75,-0.8,0]
    else:
        n = 5
        D = np.zeros((out,n))
        D[:,0] = np.array([0,0,-1,0])
        D[:,1] = np.array([-1,0,0,0])
        D[:,2] = np.array([0,1/1.2,1.25,0.75])
        D[:,3] = np.array([1,0,0,1])
        D[:,4] = np.array([0,1,0,1])
        
    D = D/np.linalg.norm(D, axis=0)
else:
    n = 2*out
    D = np.zeros((out,n))
    for o in np.arange(out):
        zeros = np.zeros(out,)
        zeros[o] = 1
        D[:,2*o] = zeros
        D[:,2*o + 1] = -zeros
    #D[:,-1] = np.ones(out,)
    D = D/np.linalg.norm(D, axis=0)


T = np.ones(n)*0.85

if encoding == 'semicircle':

    theta = np.linspace(0,np.pi,100)
    x = np.cos(theta)
    y = np.sin(theta)
    semicircle = np.vstack([x,y])

    r = np.tensordot(D.T, semicircle,axes=1)
    maxr = np.argmax(r,axis=0)

    fig, ax = plt.subplots(nrows = 1, ncols = 2)

    zeros = np.zeros(n,)

    ax[0].quiver(zeros,zeros,D[0,:],D[1,:], color=color_list, angles='xy', scale_units='xy', scale=1, width=0.02)

    for i in np.arange(n):

        cmap = colors.ListedColormap(['white', color_list[i%10]])
        bounds=[0,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        meshi = semicircle.copy()
        inactiveidxn = np.argwhere(r[i,:] < T[i])
        meshi[:,inactiveidxn] = np.nan

        r[i,:] = r[i,:] - T[i]
        r[i,r[i,:]<0] = 0

        if model == 'scn':
            notmax = np.argwhere(maxr != i)
            meshi[:,notmax] = np.nan    

            r[i,maxr != i] = 0
            r[i,maxr != i] = 0
        
        ax[0].plot(meshi[0,:],meshi[1,:], alpha=0.4, color=color_list[i%10])
        ax[1].plot(np.linspace(-1,1,100),r[i,:], color=color_list[i%10])

    ax[0].set_xlim([-1.1,1.1])
    ax[0].set_ylim([-0.1,1.1])
    ax[0].set_yticks([0,1])
    ax[0].set_xticks([-1,0,1])
    ax[1].set_xlim([-1,1])
    ax[1].set_ylim([0,np.max(r)+0.05])
    ax[1].set_xticks([-1,0,1])
    ax[0].set_aspect('equal')
    ax[1].set_aspect(3)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

elif encoding == 'sphere':
    theta = np.linspace(0,np.pi,100)
    x = np.cos(theta)
    y = np.sin(theta)
    semicircle = np.vstack([x,y])
    circle = np.hstack([semicircle, np.vstack([semicircle[0,:], -semicircle[1,:]])])

    np.random.seed(11)
    #Q = ortho_group.rvs(3)
    #Embed = Q[:,:2]
    Embed = np.array([[0,1],[1,0.5],[-1,0.5]])
    Embed = Embed/np.linalg.norm(Embed,axis=0)
    traj1 = Embed @ semicircle
    gc1 = Embed @ circle

    np.random.seed(12)
    # Q = ortho_group.rvs(3)
    # Embed = Q[:,:2]
    Embed = np.array([[-0.5,1],[0,0.5],[1,0.5]])
    Embed = Embed/np.linalg.norm(Embed,axis=0)
    traj2 = Embed @ semicircle
    gc2 = Embed @ circle

    Q = np.zeros((3,3))
    dist = np.linalg.norm(gc1[:,:,np.newaxis] - gc2[:,np.newaxis,:], axis=0)
    minidx = np.unravel_index(dist.argmin(), dist.shape)
    p = gc1[:,minidx[0]]
    Q[:,0] = np.array([-p[1]/p[0], 1, 0])
    Q[:,2] = p
    Q[:,1] = np.cross(Q[:,0],Q[:,2])
    Q = Q/np.linalg.norm(Q,axis=0)
    traj1 = Q.T @ traj1
    traj2 = Q.T @ traj2
    D = Q.T @ D

    r = 1
    u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:200j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    mesh = np.stack([x,y,z], axis=0)

    
    meshn = mesh.copy()
    meshs = mesh.copy()
    south = np.argwhere(mesh[2,:,:] < 0)
    north = np.argwhere(mesh[2,:,:] > 0)
    meshn[:,south[:,0],south[:,1]] = np.nan
    meshs[:,north[:,0],north[:,1]] = np.nan
    if gnomonic:
        xgnon, ygnon = gnomonic_proj(meshn[0,:,:], meshn[1,:,:], meshn[2,:,:])
        xgnos, ygnos = gnomonic_proj(meshs[0,:,:], meshs[1,:,:], meshs[2,:,:])

    
    # meshn = mesh
    # z = np.sqrt(1 - mesh[0,:,:]**2 - mesh[1,:,:]**2)
    # meshn = np.vstack([mesh, z[np.newaxis,:,:]])

    # meshs = mesh
    # meshs = np.vstack([mesh, -z[np.newaxis,:,:]])

    # rn = np.tensordot(D.T, meshn,axes=1)
    # rs = np.tensordot(D.T, meshs,axes=1)

    r = np.tensordot(D.T, mesh,axes=1)
    rn = np.tensordot(D.T, meshn,axes=1)
    rs = np.tensordot(D.T, meshs,axes=1)

    r1 = np.tensordot(D.T, traj1, axes=1)
    r2 = np.tensordot(D.T,traj2, axes=1)

    
    maxr = np.argmax(r,axis=0)
    maxr1 = np.argmax(r1,axis=0)
    maxr2 = np.argmax(r2,axis=0)

    
    fig, ax = plt.subplots(nrows = 2, ncols = 3)
    ax[1,0].remove()
    ax[1,1].remove()
    ax[1,2].remove()
    ax1= fig.add_subplot(2,2,3)
    ax2= fig.add_subplot(2,2,4)
    ax[0,0].remove()
    ax[0,0]= fig.add_subplot(2,3,1,projection='3d')

    zeros = np.zeros(n,)

    ax[0,0].quiver(zeros,zeros,zeros,D[0,:],D[1,:],D[2,:], color=color_list, arrow_length_ratio=0)

    for i in np.arange(n):

        cmap = colors.ListedColormap(['white', color_list[i%10]])
        bounds=[0,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        meshi = mesh.copy()
        inactiveidxn = np.argwhere(r[i,:,:] < T[i])
        meshi[:,inactiveidxn[:,0],inactiveidxn[:,1]] = np.nan

        inactiveidxn = np.argwhere(rn[i,:,:] < T[i])
        inactiveidxs = np.argwhere(rs[i,:,:] < T[i])
        if not gnomonic:
            meshni = meshn.copy()
            meshsi = meshs.copy()
            meshni[:,inactiveidxn[:,0],inactiveidxn[:,1]] = np.nan
            meshsi[:,inactiveidxs[:,0],inactiveidxs[:,1]] = np.nan
        else:
            xgnoni = xgnon.copy()
            ygnoni = ygnon.copy()
            xgnosi = xgnos.copy()
            ygnosi = ygnos.copy()

            xgnoni[inactiveidxn[:,0], inactiveidxn[:,1]] = np.nan
            xgnosi[inactiveidxs[:,0], inactiveidxs[:,1]] = np.nan
            ygnoni[inactiveidxn[:,0], inactiveidxn[:,1]] = np.nan
            ygnosi[inactiveidxs[:,0], inactiveidxs[:,1]] = np.nan


        r1[i,:] = r1[i,:] - T[i]
        r2[i,:] = r2[i,:] - T[i]

        r1[i,r1[i,:]<0] = 0
        r2[i,r2[i,:]<0] = 0

        if model == 'scn':
            notmax = np.argwhere(maxr != i)
            meshi[:,notmax[:,0],notmax[:,1]] = np.nan    

            r1[i,maxr1 != i] = 0
            r2[i,maxr2 != i] = 0
        
        ax[0,0].plot_surface(meshi[0,:,:],meshi[1,:,:],meshi[2,:,:], alpha=0.4, color=color_list[i%10])

        if model == 'ff':
            # ax[0,1].pcolormesh(meshin[0,:,:], meshin[1,:,:], meshin[2,:,:], alpha=0.5, cmap=cmap)
            # ax[0,2].pcolormesh(meshis[0,:,:], meshis[1,:,:], meshis[2,:,:], alpha=0.5, cmap=cmap)
            if not gnomonic:
                ax[0,1].scatter(meshni[0,:,:], meshni[1,:,:], s = 1.5, alpha=0.1, color=color_list[i%10])
                ax[0,2].scatter(meshsi[0,:,:], meshsi[1,:,:], s = 1.5, alpha=0.1, color=color_list[i%10])
            else:
                ax[0,1].scatter(xgnoni, ygnoni, s = 1.7, alpha=0.1, color=color_list[i%10])
                ax[0,2].scatter(xgnosi, ygnosi, s = 1.7, alpha=0.1, color=color_list[i%10])

        ax1.plot(np.linspace(-1,1,100),r1[i,:], color=color_list[i%10])
        ax2.plot(np.linspace(-1,1,100),r2[i,:], color=color_list[i%10])

    
    if model == 'scn':
        if not gnomonic:
            ax[0,1].scatter(meshn[0,:,:].reshape(-1,),meshn[1,:,:].reshape(-1,), s=1.2, c=color_list[np.argmax(rn,axis=0).reshape(-1,).astype(int) % 10])
            ax[0,2].scatter(meshs[0,:,:].reshape(-1,),meshs[1,:,:].reshape(-1,), s=1.2, c=color_list[np.argmax(rs,axis=0).reshape(-1,).astype(int) % 10])
        else:
            ax[0,1].scatter(xgnon.reshape(-1,),ygnon.reshape(-1,), s=2, c=color_list[np.argmax(rn,axis=0).reshape(-1,).astype(int) % 10])
            ax[0,2].scatter(xgnos.reshape(-1,),ygnos.reshape(-1,), s=2, c=color_list[np.argmax(rs,axis=0).reshape(-1,).astype(int) % 10])

    ax[0,0].plot(traj1[0,:],traj1[1,:],traj1[2,:],c='k',zorder=10)
    ax[0,0].plot(traj2[0,:],traj2[1,:],traj2[2,:],c='k', linestyle='dashed',zorder=10)

    traj1n = traj1[:,traj1[2,:]>0]
    traj2n = traj2[:,traj2[2,:]>0]
    if not gnomonic:
        ax[0,1].plot(traj1n[0,:],traj1n[1,:],linewidth=1,c='k')
        ax[0,1].plot(traj2n[0,:],traj2n[1,:],linewidth=1,c='k',linestyle='dashed')
    else:
        gnon1x, gnon1y = gnomonic_proj(traj1n[0,:],traj1n[1,:],traj1n[2,:])
        gnon2x, gnon2y = gnomonic_proj(traj2n[0,:],traj2n[1,:],traj2n[2,:])

        ax[0,1].plot(gnon1x,gnon1y,linewidth=1,c='k')
        ax[0,1].plot(gnon2x,gnon2y,linewidth=1,c='k',linestyle='dashed')

    traj1s = traj1[:,traj1[2,:]<0]
    traj2s = traj2[:,traj2[2,:]<0]
    if not gnomonic:
        ax[0,2].plot(traj1s[0,:],traj1s[1,:],linewidth=1,c='k')
        ax[0,2].plot(traj2s[0,:],traj2s[1,:],linewidth=1,c='k',linestyle='dashed')
    else:
        gnos1x, gnos1y = gnomonic_proj(traj1s[0,:],traj1s[1,:],traj1s[2,:])
        gnos2x, gnos2y = gnomonic_proj(traj2s[0,:],traj2s[1,:],traj2s[2,:])

        ax[0,2].plot(gnos1x,gnos1y,linewidth=1,c='k')
        ax[0,2].plot(gnos2x,gnos2y,linewidth=1,c='k',linestyle='dashed')

    ax[0,0].set_xlim3d(left=-1, right=1) 
    ax[0,0].set_ylim3d(bottom=-1, top=1) 
    ax[0,0].set_zlim3d(bottom=-1, top=1)
    ax[0,0].set_box_aspect([1.0, 1.0, 1.0])
    #ax[0,0].set_axis_off()
    # # make the panes transparent
    # ax[0,0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax[0,0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax[0,0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    # ax[0,0].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax[0,0].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax[0,0].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax[0,1].set_aspect('equal')
    ax[0,2].set_aspect('equal')
    if gnomonic:
        ax[0,1].set_xlim([-1.5,1.5])
        ax[0,1].set_ylim([-1.5,1.5])
        ax[0,2].set_xlim([-1.5,1.5])
        ax[0,2].set_ylim([-1.5,1.5])
    ax[0,1].set_yticks([-1,0,1])
    ax[0,2].set_yticks([-1,0,1])
    ax[0,1].set_xticks([-1,0,1])
    ax[0,2].set_xticks([-1,0,1])
    ax1.set_xlim([-1,1])
    ax1.set_ylim([0,np.max([r1,r2])+0.05])
    ax2.set_xlim([-1,1])
    ax2.set_ylim([0,np.max([r1,r2])+0.05])
    ax1.set_xticks([-1,0,1])
    ax2.set_xticks([-1,0,1])
    
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,2].spines['top'].set_visible(False)
    ax[0,2].spines['right'].set_visible(False)
    ax1.set_aspect(3)
    ax2.set_aspect(3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


elif encoding == 'cylinder':

    points = np.linspace(-1,1,100)
    pointssq = np.array(np.meshgrid(points, points))
    pointslin = pointssq.reshape(2,-1)

    Encoder = AngleEncoder()
    cpoints = Encoder.encode(pointslin[np.newaxis,0,:])
    cpoints = np.vstack([cpoints[0,:], pointslin[1,:], cpoints[1,:]])
    cpoints = cpoints.reshape((3,100,100))

    r = np.tensordot(D.T, cpoints, axes=1)

    maxs = np.argmax(r, axis=0)

    xaxis = np.linspace(-0.9,0.9,100)

    A = 1/2
    f = 2
    
    path1p = np.vstack([xaxis,A*np.sin(f*xaxis)-0.1])
    path2p = np.vstack([xaxis,A*np.sin(f*xaxis)+0.4])

    path1f = np.vstack([xaxis,A*np.sin(f*xaxis)-0.1])
    path2f = np.vstack([xaxis,A*np.cos(f*xaxis)+0.25])

    
    traj1p = Encoder.encode(path1p[np.newaxis,0,:])
    traj1p = np.vstack([traj1p[0,:], path1p[1,:], traj1p[1,:]])
    traj2p = Encoder.encode(path2p[np.newaxis,0,:])
    traj2p = np.vstack([traj2p[0,:], path2p[1,:], traj2p[1,:]])
    traj1f = Encoder.encode(path1f[np.newaxis,0,:])
    traj1f = np.vstack([traj1f[0,:], path1f[1,:], traj1f[1,:]])
    traj2f = Encoder.encode(path2f[np.newaxis,0,:])
    traj2f = np.vstack([traj2f[0,:], path2f[1,:], traj2f[1,:]])

    r1p = D.T @ traj1p
    r2p = D.T @ traj2p
    r1f = D.T @ traj1f
    r2f = D.T @ traj2f

    maxr1p = np.argmax(r1p,axis=0)
    maxr2p = np.argmax(r2p,axis=0)
    maxr1f = np.argmax(r1f,axis=0)
    maxr2f = np.argmax(r2f,axis=0)

    
    plt.figure() 
    
    fig, ax = plt.subplots(nrows = 2, ncols = 4)
    ax[0,0].remove()
    ax[0,0]= fig.add_subplot(2,4,1,projection='3d')
    ax[0,2].remove()
    ax[0,2]= fig.add_subplot(2,4,3,projection='3d')

    zeros = np.zeros(n,)
    ax[0,0].quiver(zeros,zeros,zeros,D[0,:],D[1,:],D[2,:], color=color_list, arrow_length_ratio=0)
    ax[0,2].quiver(zeros,zeros,zeros,D[0,:],D[1,:],D[2,:], color=color_list, arrow_length_ratio=0)

    for i in np.arange(n):

        pointsi = r[i,:].reshape((100,100))
        inactiveidx = np.argwhere(pointsi < T[i])

        meshi = cpoints.copy()
        meshi[:,inactiveidx[:,0],inactiveidx[:,1]] = np.nan
        
        cmap = colors.ListedColormap(['white', color_list[i%10]])
        bounds=[0,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        pointsi = np.flipud(pointsi)
        inactiveidx = np.argwhere(pointsi < T[i])
        balpha = 0.5
        alphas = np.ones((100,100))*balpha
        alphas[inactiveidx[:,0],inactiveidx[:,1]] = 0

        r1p[i,:] = r1p[i,:] - T[i]
        r2p[i,:] = r2p[i,:] - T[i]
        r1f[i,:] = r1f[i,:] - T[i]
        r2f[i,:] = r2f[i,:] - T[i]

        r1p[i,r1p[i,:]<0] = 0
        r2p[i,r2p[i,:]<0] = 0
        r1f[i,r1f[i,:]<0] = 0
        r2f[i,r2f[i,:]<0] = 0
        
        
        if model == 'scn':
            mask = (maxs == i).reshape((100,100))
            maska = np.flipud(mask)
            alphas = alphas*maska*(1/balpha)

            notmax = np.argwhere(np.logical_not(mask))
            meshi[:,notmax[:,0],notmax[:,1]] = np.nan  

            r1p[i,maxr1p != i] = 0
            r2p[i,maxr2p != i] = 0
            r1f[i,maxr1f != i] = 0
            r2f[i,maxr2f != i] = 0

        ax[0,0].plot_surface(meshi[0,:,:],meshi[1,:,:],meshi[2,:,:], alpha=0.4, color=color_list[i%10])
        ax[0,2].plot_surface(meshi[0,:,:],meshi[1,:,:],meshi[2,:,:], alpha=0.4, color=color_list[i%10])
       
        ax[0,1].imshow(pointsi, extent=[-1,1,-1,1], alpha=alphas, cmap=cmap, norm=norm)
        ax[0,3].imshow(pointsi, extent=[-1,1,-1,1], alpha=alphas, cmap=cmap, norm=norm)

        ax[1,0].plot(xaxis,r1p[i,:], color=color_list[i%10])
        ax[1,1].plot(xaxis,r2p[i,:], color=color_list[i%10])
        ax[1,2].plot(xaxis,r1f[i,:], color=color_list[i%10])
        ax[1,3].plot(xaxis,r2f[i,:], color=color_list[i%10])
    
    ax[0,0].plot(traj1p[0,:],traj1p[1,:],traj1p[2,:],c='k')
    ax[0,0].plot(traj2p[0,:],traj2p[1,:],traj2p[2,:],c='k', linestyle='dashed')

    ax[0,2].plot(traj1f[0,:],traj1f[1,:],traj1f[2,:],c='k')
    ax[0,2].plot(traj2f[0,:],traj2f[1,:],traj2f[2,:],c='k', linestyle='dashed')


    ax[0,1].plot(path1p[0,:], path1p[1,:], linewidth=1, c='k')
    ax[0,1].plot(path2p[0,:], path2p[1,:], linewidth=1, c='k', linestyle='dashed')

    ax[0,3].plot(path1f[0,:], path1f[1,:], linewidth=1, c='k')
    ax[0,3].plot(path2f[0,:], path2f[1,:], linewidth=1, c='k', linestyle='dashed')

    
    ax[0,0].set_xlim3d(left=-1, right=1) 
    ax[0,0].set_ylim3d(bottom=-1, top=1) 
    ax[0,0].set_zlim3d(bottom=0, top=1)
    ax[0,0].set_box_aspect([1.0, 1.0, 0.5])
    ax[0,2].set_xlim3d(left=-1, right=1) 
    ax[0,2].set_ylim3d(bottom=-1, top=1) 
    ax[0,2].set_zlim3d(bottom=0, top=1)
    ax[0,2].set_box_aspect([1.0, 1.0, 0.5])
    ax[0,1].set_aspect('equal')
    ax[0,3].set_aspect('equal')
    ax[0,1].set_xlim([-1,1])
    ax[0,1].set_ylim([-1,1])
    ax[0,3].set_xlim([-1,1])
    ax[0,3].set_ylim([-1,1])
    ax[1,0].set_xlim([-1,1])
    ax[1,0].set_ylim([0,1])
    ax[1,1].set_xlim([-1,1])
    ax[1,1].set_ylim([0,1])
    ax[1,2].set_xlim([-1,1])
    ax[1,2].set_ylim([0,1])
    ax[1,3].set_xlim([-1,1])
    ax[1,3].set_ylim([0,1])

    
else:

    points = np.linspace(-1,1,100)
    pointssq = np.array(np.meshgrid(points, points))
    pointslin = pointssq.reshape(2,-1)

    Encoder = AngleEncoder()
    tpoints = Encoder.encode(pointslin)

    r = D.T @ tpoints

    maxs = np.argmax(r, axis=0)

    u = np.mgrid[0:2 * np.pi:200j]
    x = np.cos(u)
    y = np.sin(u)

    xaxis = np.linspace(-0.9,0.9,100)

    A = 1/2
    f = 2
    path1p = np.vstack([xaxis,np.zeros_like(xaxis)-0.25])
    path2p = np.vstack([xaxis,np.zeros_like(xaxis)+0.5])

    path1f = np.vstack([xaxis,A*np.sin(f*xaxis)-0.25])
    path2f = np.vstack([xaxis,A*np.sin(f*xaxis)+0.25])

    traj1p = Encoder.encode(path1p)
    traj2p = Encoder.encode(path2p)
    traj1f = Encoder.encode(path1f)
    traj2f = Encoder.encode(path2f)

    r1p = D.T @ traj1p
    r2p = D.T @ traj2p
    r1f = D.T @ traj1f
    r2f = D.T @ traj2f

    maxr1p = np.argmax(r1p,axis=0)
    maxr2p = np.argmax(r2p,axis=0)
    maxr1f = np.argmax(r1f,axis=0)
    maxr2f = np.argmax(r2f,axis=0)

    
    plt.figure() 
    
    _, ax = plt.subplots(2,4)

    zeros = np.zeros(n,)
    circle = np.stack([x,y], axis=0)
    ax[0,0].quiver(zeros,zeros,D[0,:],D[1,:], color=color_list, angles='xy', scale_units='xy', scale=1, width=0.02)
    ax[0,0].plot(circle[0,:],circle[1,:],c='k',linestyle='dashed')
    ax[0,0].set_aspect('equal')
    ax[0,0].set_xlim([-1,1])
    ax[0,0].set_ylim([-1,1])

    ax[0,1].quiver(zeros,zeros,D[2,:],D[3,:], color=color_list, angles='xy', scale_units='xy', scale=1, width=0.02)
    ax[0,1].plot(circle[0,:],circle[1,:],c='k',linestyle='dashed')
    ax[0,1].set_aspect('equal')
    ax[0,1].set_xlim([-1,1])
    ax[0,1].set_ylim([-1,1])

    for i in np.arange(n):

        pointsi = r[i,:].reshape((100,100))
        pointsi = np.flipud(pointsi)
        inactiveidx = np.argwhere(pointsi < T[i])
        
        # For graded plotting
        # pointsi[inactiveidx[:,0],inactiveidx[:,1]] = 0
        # alphas = pointsi/np.max(pointsi) if np.max(pointsi) > 0 else np.zeros_like(pointsi)
        # alphas = 1/(1-T)*alphas - T/(1-T)
        # alphas[alphas < 0] = 0

        cmap = colors.ListedColormap(['white', color_list[i%10]])
        bounds=[0,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        balpha = 0.5
        alphas = np.ones((100,100))*balpha
        alphas[inactiveidx[:,0],inactiveidx[:,1]] = 0

        r1p[i,:] = r1p[i,:] - T[i]
        r2p[i,:] = r2p[i,:] - T[i]
        r1f[i,:] = r1f[i,:] - T[i]
        r2f[i,:] = r2f[i,:] - T[i]

        r1p[i,r1p[i,:]<0] = 0
        r2p[i,r2p[i,:]<0] = 0
        r1f[i,r1f[i,:]<0] = 0
        r2f[i,r2f[i,:]<0] = 0
        
        
        if model == 'scn':
            mask = maxs == i
            mask = np.flipud(mask.reshape((100,100)))
            alphas = alphas*mask*(1/balpha)

            r1p[i,maxr1p != i] = 0
            r2p[i,maxr2p != i] = 0
            r1f[i,maxr1f != i] = 0
            r2f[i,maxr2f != i] = 0
       
        ax[0,2].imshow(pointsi, extent=[-1,1,-1,1], alpha=alphas, cmap=cmap, norm=norm)
        ax[0,3].imshow(pointsi, extent=[-1,1,-1,1], alpha=alphas, cmap=cmap, norm=norm)

        ax[1,0].plot(xaxis,r1p[i,:], color=color_list[i%10])
        ax[1,1].plot(xaxis,r2p[i,:], color=color_list[i%10])
        ax[1,2].plot(xaxis,r1f[i,:], color=color_list[i%10])
        ax[1,3].plot(xaxis,r2f[i,:], color=color_list[i%10])
    
    ax[0,2].plot(path1p[0,:], path1p[1,:], linewidth=1, c='k')
    ax[0,2].plot(path2p[0,:], path2p[1,:], linewidth=1, c='k', linestyle='dashed')

    ax[0,3].plot(path1f[0,:], path1f[1,:], linewidth=1, c='k')
    ax[0,3].plot(path2f[0,:], path2f[1,:], linewidth=1, c='k', linestyle='dashed')

    maxt = np.max([r1p,r2p,r1f,r2f])
    ax[0,2].set_xlim([-1,1])
    ax[0,2].set_ylim([-1,1])
    ax[0,3].set_xlim([-1,1])
    ax[0,3].set_ylim([-1,1])
    [ax[1,i].set_ylim([0,maxt+0.05]) for i in np.arange(4)]
    [ax[1,i].set_xlim([-1,1]) for i in np.arange(4)]
    [ax[1,i].set_aspect(3) for i in np.arange(4)]
    [ax[1,i].spines['top'].set_visible(False) for i in np.arange(4)]
    [ax[1,i].spines['right'].set_visible(False) for i in np.arange(4)]

plt.tight_layout()
filepath = "{0}_torusvis.png".format(basepath)
plt.savefig(filepath, dpi=600, bbox_inches='tight')