# jax
from jax import jit, Array, random, config
import jax.numpy as jnp

config.update("jax_enable_x64", True)

# trapz exception
try:
    # added in jax 0.4.16
    from jax.scipy.integrate import trapezoid as trapz
except ImportError:
    # deprecated in jax 0.4.24
    from jax.numpy import trapz

# jax_cosmo
from jax_cosmo import Cosmology
from jax_cosmo.angular_cl import angular_cl, noise_cl
from jax_cosmo.probes import WeakLensing
from jax_cosmo.redshift import smail_nz

# tinygp
import tinygp

import J0_jax
import mcfit_jax as mcfit
import grftools_jax as grftools

###################################################################
# HELPER FUNCTIONS ################################################
###################################################################

def getgrid(px):
    """
    Generate a 2D grid.

    Args:
    px (int): Discrete size of the map.
    """
    x_grid, y_grid = jnp.arange(px, dtype=jnp.float64), jnp.arange(px, dtype=jnp.float64)
    x_, y_ = jnp.meshgrid(x_grid, y_grid)
    X_pred = jnp.vstack((x_.flatten(), y_.flatten())).T
    return X_pred, x_, y_


def radial_profile(px, step=1):
    """
    Calculate radial distance from center of a map.

    Args:
    px (int): Discrete size of the map.
    step (int): Step size for radial profile.
    """
    center = px // 2
    y, x = jnp.abs((jnp.indices((px, px)) - center)) * step
    r = jnp.sqrt(x**2 + y**2)
    return r


def param_to_cosmo(p):
    """
    Convert parameters to a cosmology object.

    Args:
    p (dict): cosmological parameters.
    """
    # Fiducial values in case of missing parameters.
    fiducialp = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "Omega_k": 0.0,
        "h": 0.7,
        "n_s": 0.97,
        "sigma8": 0.8,
        "S8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
    }

    # Overwrite fiducial with given parameters.
    for key in p:
        fiducialp[key] = p[key]
    
    # Make jaxcosmo cosmology.
    if "S8" in p and "sigma8" in p:
        Omega_m = 0.3*(fiducialp["S8"]/fiducialp["sigma8"])**2
        cosmo = Cosmology(
            Omega_c=(Omega_m - fiducialp["Omega_b"]),
            Omega_b=fiducialp["Omega_b"],
            Omega_k=fiducialp["Omega_k"],
            h=fiducialp["h"],
            n_s=fiducialp["n_s"],
            sigma8=(fiducialp["sigma8"]),
            w0=fiducialp["w0"],
            wa=fiducialp["wa"],
        )
    elif "S8" in p and "Omega_m" in p:
        cosmo = Cosmology(
            Omega_c=(fiducialp["Omega_m"] - fiducialp["Omega_b"]),
            Omega_b=fiducialp["Omega_b"],
            Omega_k=fiducialp["Omega_k"],
            h=fiducialp["h"],
            n_s=fiducialp["n_s"],
            sigma8=(fiducialp["S8"]/jnp.sqrt(fiducialp["Omega_m"]/0.3)),
            w0=fiducialp["w0"],
            wa=fiducialp["wa"],
        )
    elif "sigma8" in p and "Omega_m" in p:
        cosmo = Cosmology(
            Omega_c=(fiducialp["Omega_m"] - fiducialp["Omega_b"]),
            Omega_b=fiducialp["Omega_b"],
            Omega_k=fiducialp["Omega_k"],
            h=fiducialp["h"],
            n_s=fiducialp["n_s"],
            sigma8=fiducialp["sigma8"],
            w0=fiducialp["w0"],
            wa=fiducialp["wa"],
        )
    return cosmo


def LNk(x, alpha):
    """
    Map lognormal transformation.

    Args:
    x (Array): Map (k) values.
    alpha (float): Lognormal shift parameter.
    """
    return alpha * (jnp.exp(x - 0.5 * jnp.std(x) ** 2) - 1)


def iLNw(x, alpha):
    """
    Correlation function inverse lognormal transformation.

    Args:
    x (Array): Correlation function (w) values.
    alpha (float): Lognormal shift parameter.
    """
    return jnp.log(1 + x / alpha / alpha)


def makeellrange(px, L):
    """
    Make range of ell values.

    Args:
    px (int): Discrete size of the map.
    L (float): Physical size of the map.
    """
    ellmin = 2 * jnp.pi / L
    ellmax = 2 * jnp.pi * px / L
    return ellmin, ellmax


def makethetarange(px, L):
    """
    Make range of theta values.

    Args:
    px (int): Discrete size of the map.
    L (float): Physical size of the map.
    """
    thetamin = 0
    thetamax = L * jnp.sqrt(2)
    return thetamin, thetamax


###################################################################
# DISCRETE POWER SPECTRUM #########################################
###################################################################

class DiscreteCl:
    """
    Class for handling discrete cosmological power spectrum (Cl).
    """

    def __init__(self, p, px, L, ellrange=None, filter=True):
        """
        Initialize the DiscreteCl object.

        Args:
        p (dict): Cosmological parameters.
        px (int): Discrete size of the map.
        L (float): Physical size of the map.
        ellrange (tuple): Range of ell values.
        filter (bool): Whether to apply sinc filter to Cl.
        """
        self.px = px
        self.L = L

        # Compute Cl
        if ellrange is None:
            self.ellmin, self.ellmax = makeellrange(px, L)
        else:
            self.ellmin, self.ellmax = ellrange[0], ellrange[1]

        # Compute from theory
        self.ell = jnp.logspace(jnp.log10(self.ellmin), jnp.log10(self.ellmax), num=50)
        self.clarr, self.cl = self.getcl(p, filter)

    def getcl(self, p, filter):
        """
        Compute the theoretical Cl.

        Args:
        p (dict): Cosmological parameters.
        filter (bool): Whether to apply sinc filter to Cl.
        """
        # Define the redshift bin
        a, b, z0 = 3.5, 4.5, 1.0   # bin 5 KiDS-1000
        probe = WeakLensing([smail_nz(a, b, z0)], sigma_e=0)

        cls = angular_cl(param_to_cosmo(p), self.ell, [probe])
        clarr = cls[0]

        # Apply filter if required
        if filter:
            sincfilter = lambda x: jnp.sinc(x / (2 * jnp.pi * self.px / self.L))
            sincfilterarr = sincfilter(self.ell)
            clarr = clarr * sincfilterarr**4

            clinterp = lambda x: jnp.interp(x, self.ell, clarr)
            atol = 1e-8
            cl = lambda x: jnp.where((x >= self.ellmin - atol) & (x <= self.ellmax + atol), clinterp(x), 0.0)
        else:
            cl = lambda x: jnp.interp(x, self.ell, clarr)
        return clarr, cl

    def getdcl(self, clpx, step):
        """
        Get the discrete Cl.

        Args:
        clpx (int): Discrete size of the wanted power spectrum.
        step (float): Step size for radial profile.
        """
        ell_2d = 2 * jnp.pi * radial_profile(clpx, step) / self.L
        return self.cl(ell_2d)

    def getgrf(self, PRNGKey1, PRNGKey2):
        """
        Generate a Gaussian random field (GRF).

        Args:
        PRNGKey1, PRNGKey2 (type(jax.random.PRNGKey)): Random number generator keys.
        """
        grf = grftools.pseudo_Cls.create_Gaussian_field(PRNGKey1, PRNGKey2, self.cl, (self.px, self.px), (self.L, self.L))
        return grf

###################################################################
# CORRELATION FUNCTION FROM POWER SPECTRUM ########################
###################################################################

def getw_integration(ell, clarr, L):
    """
    Perform an integration over a function involving the Bessel function of the first kind.

    Args:
    ell (Array): Ell values of the corresponding clarr.
    clarr (Array): 1D array of the power spectrum.
    L (float): Physical size of the map.

    Returns:
    tuple: A tuple containing the theta values and the result of the integration.
    """
    # Create a finer grid for ell
    ellfine = jnp.linspace(ell.min(), ell.max(), num=248)
    # Interpolate clarr onto the finer grid
    clarr = jnp.interp(ellfine, ell, clarr)

    # Define the function to be integrated
    def argumentarr(theta):
        return ellfine * clarr * J0_jax.J0(ellfine * theta) / 2 / jnp.pi

    # Define the integration operation
    def wtrapz(theta):
        return trapz(argumentarr(theta), ellfine)

    # Create a theta grid consisting of a linear part and a logarithmic part
    mid = L * jnp.sqrt(2) / 1000
    thetalin = jnp.linspace(0, mid, num=16)
    thetalog = jnp.logspace(jnp.log10(mid), jnp.log10(L * jnp.sqrt(2)), num=48)
    theta = jnp.append(thetalin, thetalog)

    # Perform the integration for each theta value
    warr = jnp.vectorize(wtrapz)(theta)

    return theta, warr

def getw_FFTlog(ell, clarr, px, L, filter=False, lowring=False):
    """
    Perform an inverse Fast Fourier Transform (FFT) on the input clarr.

    Args:
    ell (Array): Ell values of the corresponding clarr.
    clarr (Array): 1D array of the power spectrum.
    px (int): Discrete size of the map.
    L (float): Physical size of the map.
    filter (bool, optional): Whether to apply sinc filter to Cl.
    lowring (bool, optional): Whether to apply the lowring option in mcfit.

    Returns:
    tuple: A tuple containing the theta values and the result of the FFT.
    """
    # Extend the ell grid
    ellmin, ellmax = 2*jnp.pi / L / 4, 2*jnp.pi / L * px
    ellfine = jnp.logspace(
        jnp.log10(ellmin), jnp.log10(ellmax), num=1024, endpoint=False
    )
    # Define the interpolation function
    log_cl = lambda x: jnp.interp(x, jnp.log(ell), jnp.log(clarr))
    cl = lambda x: jnp.exp(log_cl(jnp.log(x)))

    # Perform the inverse FFT
    theta, warr = mcfit.C2w(ell, lowring=lowring, backend="jax")(
        clarr
    )

    return theta, warr

def getw_FFT(dcl, L):
    """
    Perform an inverse Discrete Fourier Transform (DFT) on the input dcl.

    Args:
    dcl (Array): 2D array of the power spectrum.
    L (float): Physical size of the map.

    Returns:
    tuple: A tuple containing the unshifted and shifted result of the DFT.
    """
    # Perform the inverse DFT
    udw = jnp.abs(jnp.fft.ifft2(dcl))

    # Convert the inverse DFT to an inverse FT
    udw = udw * dcl.size / L**2

    # Shift the result
    dw = jnp.fft.fftshift(udw)

    return udw, dw

def getcl_FFT(dw, L):
    """
    Perform a Discrete Fourier Transform (DFT) on the input dw.

    Args:
    dw (Array): 2D array of the correlation function.
    L (float): Physical size of the map.

    Returns:
    tuple: A tuple containing the unshifted and shifted result of the DFT.
    """
    # Perform the DFT
    dcl_notshifted = jnp.abs(jnp.fft.fft2(dw))

    # Convert the DFT to an FT
    dcl_notshifted = dcl_notshifted * L**2 / dw.size

    # Shift the result
    dcl = jnp.fft.fftshift(dcl_notshifted)

    return dcl_notshifted, dcl

###################################################################
# RADIAL AVERAGING & BINNING ######################################
###################################################################

def dw2w(dw, px, L, step=1, callable=False):
    """
    Convert a 2D correlation function to 1D with a radial profile.

    Args:
    dw (Array): 2D correlation function.
    px (int): Discrete size of the map.
    L (float): physical Size of the map.
    step (int, optional): Step size for radial profile.
    callable (bool, optional): If True, returns an interpolated function.

    Returns:
    tuple: A tuple containing theta values and the result of the radial averaging.
    """
    # Create a radial profile
    rbin = radial_profile(px, step).astype("int32")

    # Perform radial averaging
    nbins = px // 2
    tbin = jnp.bincount(rbin.ravel(), dw.ravel(), length=nbins)
    nr = jnp.bincount(rbin.ravel(), length=nbins)
    radialprofile = tbin / nr

    warr = radialprofile

    # Convert rbin to theta
    pixel_size = L / px
    rbin = jnp.arange(nbins)
    theta = rbin * pixel_size  # [rad]

    thetamin, thetamax = makethetarange(px, L)

    # Interpolation
    if callable:
        winterp = lambda x: jnp.interp(x, theta, warr)
        return theta, winterp
    else:
        return theta, warr


def dcl2cl(dcl, px, L, step=1, callable=False):
    """
    Convert a 2D power spectrum to 1D with a radial profile.

    Args:
    dcl (Array): 2D power spectrum.
    px (int): Discrete size of the map.
    L (float): Physical size of the map.
    step (int, optional): Step size for radial profile.
    callable (bool, optional): If True, returns an interpolated function.

    Returns:
    tuple: A tuple containing theta values and the result of the radial averaging.
    """
    # Create a radial profile
    rbin = radial_profile(px, step).astype("int32")

    # Perform radial averaging
    nbins = px // 2
    tbin = jnp.bincount(rbin.ravel(), dcl.ravel(), length=nbins)
    nr = jnp.bincount(rbin.ravel(), length=nbins)
    radialprofile = tbin / nr

    clarr = radialprofile

    # Convert rbin to ell
    pixel_size = 2 * jnp.pi / L
    rbin = jnp.arange(nbins)
    ell = rbin * pixel_size  # [rad]

    ellmin, ellmax = makeellrange(px, L)

    # Interpolation
    if callable:
        clinterp = lambda x: jnp.interp(x, ell, clarr)
        return ell, clinterp
    else:
        return ell, clarr

###################################################################
# CUSTOM TINYGP KERNELS ###########################################
###################################################################

class Kernel1D(tinygp.kernels.Kernel):
    """
    A custom kernel class for a tinygp.kernels.Kernel given a 1D correlation function.
    """
    res: Array
    theta: Array
    warr: Array

    def __init__(self, theta, warr, px, L):
        """
        Initialize the kernel with given theta, warr, px and L.

        Args:
        theta (Array): Theta values of the corresponding warr.
        warr (Array): 1D array of the correlation function.
        px (int): Discrete size of the map.
        L (float): Physical size of the map.
        """
        self.theta, self.warr = theta, warr
        self.res = L / px  # res (Array): Resolution of the field.

    def evaluate(self, X1, X2):
        """
        Evaluate the kernel at given points X1 and X2.

        Args:
        X1, X2 (Array): 2D points on map.

        Returns:
        float: The kernel evaluated at physical distance between X1 and X2.
        """
        atol = 1e-8
        w = lambda x: jnp.where(
            (x >= 0 - atol) & (x <= self.theta.max() + atol),
            jnp.interp(x, self.theta, self.warr),
            0.0,
        )
        tau = jnp.sqrt(jnp.sum((X1 - X2) ** 2)) * self.res
        result = w(tau)
        return result
    
    def getw(self):
        """
        Get the radial distances and power spectrum values.

        Returns:
        tuple: A tuple containing theta and correlation function values.
        """
        return self.theta, self.warr
    
class Kernel2D(tinygp.kernels.Kernel):
    """
    A custom kernel class for a tinygp.kernels.Kernel given a 2D correlation function.
    """
    udw: Array
    r: Array

    def __init__(self, udw, px):
        """
        Initialize the kernel with given udw and px.

        Args:
        udw (Array): Unshifted 2D correlation function resulting from FFT of power spectrum.
        px (int): Discrete size of the map.
        """
        self.udw = udw
        self.r = jnp.sqrt(self.udw.size / px**2)  # r (Array): Ratio of the size of the 2D power spectrum to the number of pixels in the map squared.

    def evaluate(self, X1, X2):
        """
        Evaluate the kernel at given points X1 and X2.

        Args:
        X1, X2 (Array): Points at which to evaluate the kernel.

        Returns:
        float: 2D correlation between X1 and X2.
        """
        tau0 = (jnp.abs(X1[0] - X2[0]) * self.r).astype("int32")
        tau1 = (jnp.abs(X1[1] - X2[1]) * self.r).astype("int32")
        result = self.udw[tau1][tau0]
        return result

    def dw2w(self, px, L, step=1, callable=False):
        """
        Convert a 2D correlation function to 1D with a radial profile with the help of dw2w.
        """
        return dw2w(jnp.fft.fftshift(self.udw), px, L, step, callable)

###################################################################
# DATA MANIPULATION ###############################################
###################################################################


def makenoise(PRNGKey, noise_amplitude, px):
    """
    Generate noise with a normal distribution.

    Args:
    PRNGKey (type(jax.random.PRNGKey)): Random number generator key.
    noise_amplitude (float): Amplitude of the noise.
    px (int): Discrete size of the map.

    Returns:
    Array: Noise array of shape (px, px).
    """
    noise = noise_amplitude * random.normal(PRNGKey, shape=(px, px))
    return noise


def save_data(data, filename="data.npy"):
    """
    JAX compatible way of saving data to a file.

    Args:
    data (Array): Data to be saved.
    filename (str, optional): Name of the file. Default is "data.npy".
    """
    with open(filename, "wb") as f:
        jnp.save(f, data)


def getmask(PRNGKey, px, fsky=0.5, mask_id="random"):
    """
    Generate a mask.
    4 options for mask generation, "block", "tri", "random", and a number (a mask made of number small squares).
    "block" leaves a block of unmasked pixels in the center of the map.
    "random" is a random mask with a fraction of the sky to be masked.
    "tri" is three blocks with three different fractions of random masks: 0.95, 0.5, and 0.05.
    (int) creates a number of small squares distributed randomly in the map.
    
    Args:
    PRNGKey (Array): Random number generator key.
    px (int): Discrete size of the map.
    fsky (float, optional): Fraction of the sky to be masked. Default is 0.5.
    mask_id (str or int, optional): Type of mask. Default is "random".

    Returns:
    Array: Mask, array of indeces.
    """
    def shuffle(start, stop):
        foo = random.permutation(PRNGKey, jnp.arange(start, stop), independent=True)
        return foo

    tot = px * px

    if mask_id == "block":
        halffskytot = int(tot * fsky / 2)
        N1 = jnp.arange(0, halffskytot)
        N2 = jnp.arange(tot - halffskytot, tot)
        N = jnp.append(N1, N2)
    elif mask_id == "random":
        N = shuffle(0, tot)[: int(fsky * tot)]
    elif mask_id == "tri":
        fifthtot = int(tot / 5)
        N1 = shuffle(0, fifthtot)[: int(0.95 * fifthtot)]
        N2 = shuffle(fifthtot, 4 * fifthtot)[: int(0.5 * 3 * fifthtot)]
        N3 = shuffle(4 * fifthtot, tot)[: int(0.05 * fifthtot)]
        N = jnp.append(N1, jnp.append(N2, N3))
    elif mask_id >= 1:
        length = px // 10
        Nholes = shuffle(0, tot - px * length)[: int(mask_id)]

        def buildrow(sN, length):
            Nholerow = jnp.array([sN])
            for i in range(1, length):
                Nholerow = jnp.append(Nholerow, jnp.array([sN + i]))
            return Nholerow

        def buildblock(sN, length):
            Nholeblock = buildrow(sN, length)
            for i in range(length - 1):
                Nholeblock = jnp.append(
                    Nholeblock,
                    buildrow(Nholeblock[length * i] + px, length),
                )
            return Nholeblock

        Nblocks = jnp.array([buildblock(Nhole, length) for Nhole in Nholes]).flatten()
        from numpy import setdiff1d

        N = setdiff1d(jnp.arange(0, tot), Nblocks)
    return jnp.sort(N)


def complete(x, mask, shape):
    """
    Useful function to plot masked array, substitutes a masked index with a jnp.nan.

    Args:
    x (Array): Array to be completed.
    mask (Array): Mask.
    shape (tuple): Shape of the wanted completed array (px, px).

    Returns:
    Array: Completed array.
    """
    if x.size == shape[0] * shape[1]:
        return x.reshape(shape)
    else:
        completex = jnp.array([jnp.nan for _ in range(shape[0] * shape[1])])
        completex = completex.at[mask.flatten()].set(x.flatten())
        return completex.reshape(shape)


###################################################################
# END CODE ########################################################
###################################################################