# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

def hybrid_bkg(data):
    """Background estimator used in SExtractor.

    Parameters
    ----------
    data : array_like
        The data for which to estimate the background.

    Returns
    -------
    bkg : float
        Estimate of background of the data.
    std : float
        Standard deviation of clipped data
    """

    from astropy import tools

    # Sigma-clip the data at +/-3 sigma around median until convergence.
    # iters = None takes forever
    data = np.asarray(data)
    filtered_data, mask = tools.sigma_clip(data, sig=3., cenfunc=np.median,
                                           iters=4)

    # If the standard deviation changed less than 20% during the
    # clipping process, consider the field to be uncrowded and simply
    # use mean of the clipped distribution. Otherwise, use an estimate
    # of the mode.
    raw_std = np.std(data)
    filtered_std = np.std(filtered_data)
    if abs((filtered_std - raw_std) / raw_std) < 0.2:
        return np.mean(filtered_data), filtered_std
    else:
        filtered_mode = (2.5 * np.median(filtered_data) - 
                         1.5 * np.mean(filtered_data))
        return filtered_mode, filtered_std


def clipmode_bkg(data):

    from astropy import tools

    data = np.asarray(data)
    filtered_data, mask = tools.sigma_clip(data, sig=3., cenfunc=np.median,
                                           iters=3)
    filtered_std = np.std(filtered_data)
    filtered_mode = (2.5 * np.median(filtered_data) - 
                     1.5 * np.mean(filtered_data))
    return filtered_mode, filtered_std


def background(data, mask=None, boxsize=None, boxadjust='even',
               bkgfunc=clipmode_bkg, min_unmasked_frac=0.5):

    """Estimate global background and standard deviation of image data.

    Currently, only 2-d arrays are supported.
    
    Parameters
    ----------
    data : array_like
        The array on which to estimate the background.
    boxsize : int, optional
        The size of boxes in pixels. Can be a tuple or array (height,
        width) for rectangles. If None, a scalar (non-variable)
        background is returned.
    box_adjust : {'even', 'last', 'none'}
        How to adjust boxes if the data shape is not an exact multiple of
        boxsize:
        
        'even'
            Evenly space all boxes, making them as close to `boxsize` as
            possible.
        'last'
            Adjust last box: if less than half of `boxsize`, combine with
            previous box. Otherwise leave as own box.
        'none'
            No adjustment. Last box will be datashape % boxsize.

    method : {'mean', 'median', 'mode', 'clipmean',
        'clipmedian', 'clipmode', 'hybrid'}, optional
        Method used to estimate background in each box.
    mode_estimator: sequence, optional
        For methods that involve the mode, the mode is estimated with
        mode = mode_estimator[0] * median - mode_estimator[1] * mean
    smoothing: {''}
        How to interpolate between boxes. If ``None``, do not interpolate.
    mask : array_like, bool, optional
        Pixels to ignore when estimating background. Must be same shape
        as `data`.

    Returns
    -------
    bkg : float or `~numpy.ndarray`
    std : float or `~numpy.ndarray`
    """

    from scipy.interpolate import griddata

    # Check dimensionality of data
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("only 2-d arrays supported.")

    # Check that mask exists and is of the correct shape.
    if mask is not None:
        mask = np.asarray(mask).astype(np.bool)
        if mask.shape != data.shape:
            raise ValueError("shapes of mask and data must match")

    # If boxsize is None, just go ahead and return scalars
    if boxsize is None:
        if mask is None:
            boxdata = data
        else:
            boxdata = data[mask]
        return bkgfunc(boxdata)

    # Make boxsize a 1-d array of ints.
    boxsize = np.atleast_1d(boxsize).astype(np.int)
    if boxsize.ndim != 1:
        raise ValueError("Boxsize can be at most 1-dimensional")

    # Broadcast boxsize to match data.shape.
    # (`datashape` is not used, this is just for broadcasting boxsize.)
    if boxsize.shape[0] not in [1, 2]:
        raise ValueError("boxsize must be length 1 or 2.")
    datashape, boxsize = np.broadcast_arrays(data.shape, boxsize)
    
    # Define bin_edges
    bin_edges = []
    if boxadjust == 'none':
        for i in [0, 1]:
            bin_edges.append(
                np.arange(0, data.shape[i] + boxsize[i], boxsize[i]))
            bin_edges[i][-1] = data.shape[i]

    elif boxadjust == 'last':
        for i in [0, 1]:
            remainder = data.shape[i] % boxsize[i]
            if remainder == 0 or remainder > boxsize[i] / 2.:
                bin_edges.append(np.arange(0, data.shape[i] + boxsize[i],
                                           boxsize[i]))
            else:
                bin_edges.append(np.arange(0, data.shape[i], boxsize[i]))
            bin_edges[i][-1] = data.shape[i]
    elif boxadjust == 'even':
        for i in [0, 1]:
            n_boxes = int(data.shape[i] / boxsize[i] + 0.5)
            bin_edges.append(
                np.linspace(0, data.shape[i], n_boxes + 1).astype('int32'))
    else:
        raise ValueError('Unrecognized value for boxadjust: '
                         '{0}'.format(boxadjust))

    # Number of boxes in y and x.
    ny = bin_edges[0].shape[0] - 1
    nx = bin_edges[1].shape[0] - 1

    # Initialize arrays to hold result in each box.
    mesh_bkg = np.empty((ny, nx), dtype=np.float)
    mesh_std = np.empty((ny, nx), dtype=np.float)
    mesh_ok = np.ones((ny, nx), dtype=np.bool)  # whether there is "enough"
                                                # data in each box.

    # Loop over boxes
    for j in range(ny):
        for i in range(nx):

            # Slice defining the box.
            boxslice = [slice(bin_edges[0][j], bin_edges[0][j + 1]),
                        slice(bin_edges[1][i], bin_edges[1][i + 1])]
            subdata = data[boxslice]

            if mask is not None:
                submask = mask[boxslice]

                # Are there enough unmasked pixels in the box to compute
                # a good value?
                total_boxpix = ((bin_edges[0][j + 1] - bin_edges[0][j]) *
                                (bin_edges[1][i + 1] - bin_edges[1][i]))
                boxok = np.sum(submask) / total_boxpix > min_unmasked_frac

                if not boxok:  # If not, then flag this box and move on.
                    mesh_ok[j, i] = False
                    continue

                subdata = subdata[submask]  # If ok, then use good pixels.

            # Get background, standard deviation estimate for this box
            box_bkg, box_std = bkgfunc(subdata)
            mesh_bkg[j, i] = box_bkg
            mesh_std[j, i] = box_std

    ctr_y = 0.5 * (bin_edges[0][:-1] + bin_edges[0][1:])  # box centers in x
    ctr_x = 0.5 * (bin_edges[1][:-1] + bin_edges[1][1:])  # box centers in y
    CTR_X, CTR_Y = np.meshgrid(ctr_x, ctr_y)  # 2-d array of box centers
    
    # Reshape everything into 1-d arrays, only including valid box centers
    boxctrx = CTR_X[mesh_ok]
    boxctry = CTR_Y[mesh_ok]
    bkgvals = mesh_bkg[mesh_ok]
    stdvals = mesh_bkg[mesh_ok]

    # Define output X, Y coordinates:
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    bkg = griddata((boxctrx, boxctry), bkgvals, (X, Y), method='cubic')
    std = griddata((boxctrx, boxctry), stdvals, (X, Y), method='cubic')

    return bkg, std
