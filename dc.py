from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import proxmin

# Reference points for different spectra
ref_points = OrderedDict([
    ('water', (105, 40)),
    #('dirt', (84, 102)),
    #('dirt', (43, 23)),
    ('dirt', (145, 21)),
    ('grass', (104, 105)),
    ('roof', (68, 170)),
    #('concrete', (94, 170)),
    #('concrete', (105, 23)),
    ('concrete', (102, 128)),
    ('trees', (187, 95)),
    ('road', (76, 86)),
    #('road', (113, 196)),
    #('shadow', (18, 160))
    ('shadow', (20, 142))
])

monument_points = OrderedDict([
    ("concrete", (162, 507)), # statue
        #("tar", (175, 546)), # road, degenerate with concrete
        #("roof2", (114, 575)), # roof, degenerate with dirt
        ("dirt", (172, 597)), # dirt
        ("grass", (125, 575)), #grass
        #("trees", (183, 561)), #tree
        ("bkg", (None,None)
    )
])

# Color associated with each reference object
ref_colors = {
    'water': '#0000ff',
    'grass': '#00ff00',
    'roof': '#07afc1',
    'roof2': '#bf4101',
    'concrete': '#e0e0e0',
    'dirt': '#7c600b',
    'trees': '#2c702c',
    'bkg': '#000000',
    'shadow': '#000000'
}

def plot_color_img(data, shape, figsize=(8,8), ax=None, bounds=None, show=True):
    """Plot a color image of hyperspectral data
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)

    # Convert spectra to RGB colors
    if bounds is not None:
        r,g,b = bounds(data, shape)
    else:
        # Map spectra to RGB colors
        b = np.sum(data[:50], axis=0).reshape(shape)
        g = np.sum(data[50:100], axis=0).reshape(shape)
        r = np.sum(data[100:], axis=0).reshape(shape)
        # Set on a scale from 0-255
        b = b-np.min(b)
        g = g-np.min(g)
        r = r-np.min(r)

        b = b/np.max(b)
        g = g/np.max(g)
        r = r/np.max(r)
    # create RGB array
    rgb = (np.dstack((r,g,b)) * 255.999) .astype(np.uint8)

    ax.imshow(rgb)

    # Display the image
    if show:
        plt.show()
    return rgb

def get_point_spec(x, y, data, img_shape):
    """Get the spectrum at a given point
    """
    #return data[:,img_shape[1]*y+x]
    # TODO: choose either the previous or the following code
    avg = [data[:,img_shape[1]*y+x]]
    if x>0:
        avg.append(data[:,img_shape[1]*y+(x-1)])
    if y>0:
        avg.append(data[:,img_shape[1]*(y-1)+x])
    if x<img_shape[1]:
        avg.append(data[:,img_shape[1]*y+(x+1)])
    if y<img_shape[0]:
        avg.append(data[:,img_shape[1]*(y+1)+x])
    avg = np.array(avg)
    return np.mean(avg, axis=0)

def prox_ones(X, step):
    """Set all values to one
    """
    return np.ones_like(X)

def prox_bkg(X, step, bidx=-1):
    """Use a constant background
    """
    X[bidx] = np.mean(X[bidx])
    return X

def prox_bkg_plus(X, step, bidx=-1):
    """Set one component to be a flat background
    """
    X = prox_bkg(X, step, bidx)
    X = proxmin.operators.prox_plus(X, step)
    return X

def init_nmf(img, img_shape, points=None, spec=None, s_iter=4, show=True, bidx=None, features=None):
    """Initialize A0 and S0 using the reference points
    """
    # Set the proximal operator for the S update
    if bidx is not None:
        prox_g = partial(prox_bkg_plus, bidx=bidx)
    else:
        prox_g = proxmin.operators.prox_plus

    if points is not None and spec is not None:
        features = len(points)
        A0 = np.zeros((img.shape[0], features))
        S0 = np.zeros((features, img_shape[0]*img_shape[1]))

        # Use the reference points to initialize A
        for idx, (obj,(x,y)) in enumerate(points.items()):
            A0[:, idx] = spec[obj]
    elif features is None:
        raise Exception("Must either supply 'points; and 'spec' or 'features' to initialize A")
    else:
        A0 = np.random.rand(img.shape[0], features)
        S0 = np.zeros((features, img_shape[0]*img_shape[1]))

    # Normalize A0
    norm = np.sum(A0, axis=0)
    A0 = A0/norm

    # Initialize S0 by performing a few proximal updates
    step = .5/proxmin.utils.get_spectral_norm(A0)
    for n in range(4):
        S0 = proxmin.nmf.prox_likelihood_S(S0, step, A0, img, prox_g)
    
    # Plot the initial Spectra
    if show and points is not None:
        for idx, (obj,(x,y)) in enumerate(points.items()):
            plt.plot(A0[:, idx], label=obj)
        plt.legend()
        plt.show()
        for idx, (obj, pt) in enumerate(points.items()):
            plt.imshow(S0[idx].reshape(img_shape), cmap="Greys_r")
            plt.title(obj)
            plt.colorbar()
            plt.show()
    return A0, S0

def plot_spectra(wavelength, A, points, figsize=(12,8), ax=None, show=True):
    """Plot the spectra for each object
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
    for row in range(A.shape[1]):
        ax.plot(wavelength, A[:,row], label=list(points.keys())[row])
    if show:
        plt.legend()
        plt.show()

def plot_single_object(data, Ak, Sk, img_shape, spectra, figsize=(16,8),
                       obj=None, pt=None, color=None, wavelength=None,
                       Smax_k=None):
    """Plot the image and spectra of a single object
    """
    fig = plt.figure(figsize=figsize)

    if color is None:
        color='k'
    
    # Continuus image
    ax1 = fig.add_subplot(2,2,1)
    if obj is not None:
        ax1.set_title(obj, color=color)
    img = Sk.reshape(img_shape)
    img_plot = ax1.imshow(img, cmap="Greys_r")
    fig.colorbar(img_plot, ax=ax1)

    # Likelihood
    if Smax_k is not None:
        ax4 = fig.add_subplot(2,2,3)
        ax4.set_title(obj, color=color)
        ax4.imshow(Smax_k, cmap="Greys_r")

    # Model Spectra
    ax2 = fig.add_subplot(2,2,2)
    if wavelength is not None:
        ax2.plot(wavelength, Ak, color=color)
    else:
        ax2.plot(Ak, color=color)

    #Pixel Spectra
    if pt is not None:
        x,y = pt
        ax3 = fig.add_subplot(2,2,4)    
        if wavelength is not None:
            ax3.plot(wavelength, spectra[obj], color=color, alpha=.6)
        else:
            ax3.plot(spec, color=color, alpha=.6)
        ax3.set_title("pixel data")

    plt.show()

def plot_objects(data, A, S, img_shape, points, spectra, figsize=(16,12), wavelength=None):
    """Plot the image and spectra of each classification
    """

    # Get the color_cycle used by matplotlib so that
    # `plot_spectra` and these plots use the same colors for the
    # same objects
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Get the largest contributor to the spectrum for each pixel
    Smax = np.argmax(S, axis=0).reshape(img_shape)

    for idx, (obj,(x,y)) in enumerate(points.items()):
        Smax_k = np.zeros_like(Smax)
        Smax_k[Smax==idx] = 1
        plot_single_object(data, A[:,idx], S[idx], img_shape, spectra,
                           figsize=figsize, obj=obj, pt=(x,y), color=color_cycle[idx],
                           wavelength=wavelength, Smax_k=Smax_k)

def plot_likelihood(S, img_shape, points, ax=None, figsize=(12,10), colors=None, show=True, fig=None):
    """Plot the most likely type of object for each pixel
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)

    if colors is None:
        colors = ref_colors
    cmaplist = [colors[obj] for obj in list(points.keys())]
    cmap = mpl_colors.ListedColormap(cmaplist)

    Smax = np.argmax(S, axis=0)
    img = ax.imshow(Smax.reshape(img_shape), cmap=cmap, vmin=0, vmax=len(list(points.keys()))-1)

    if fig is not None:
        cbar = fig.colorbar(img, ax=ax)
        cbar.ax.set_yticklabels(list(points.keys()))
    
    if show:
        plt.show()

def compare_likelihood(data, img_shape, S, points, figsize=(12,6), colors=None):
    """Plot the likelihood and color image side by side
    """
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1,2,1)
    plot_color_img(data, img_shape, ax=ax1, show=False)
    
    ax2 = fig.add_subplot(1,2,2)
    plot_likelihood(S, img_shape, points, ax=ax2, fig=fig, colors=colors)
    plt.show()
