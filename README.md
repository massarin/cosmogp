# cosmogp

### Overview

`cosmogp` is a Python package designed for field-level inference of 2D weak lensing convergence fields using Gaussian processes. It includes functionalities for generating Gaussian random fields (GRF) and lognormal fields, converting between power spectra and correlation functions, and setting up `numpyro` models for MCMC simulations to derive cosmological parameter posteriors. The package builds upon `jax-cosmo` for power spectrum generation and `tinygp` for Gaussian processes.

### Features

- **Field Generation:**
  - Gaussian Random Fields (GRF)
  - Lognormal Fields

- **2-pt Statistics Conversion Routines:**
  - FFT2 for 2D Fourier transforms
  - FFTlog for Hankel transforms
  - Bessel function integration for Hankel transforms

- **Cosmological Parameter Inference:**
  - `numpyro` model setup for MCMC simulation
  - Posterior distributions for cosmological parameters such as $S_8$, $\Omega_m$, $\sigma_8$


### Installation

To install `cosmogp`, clone the repository and install the dependencies:

```sh
git clone https://github.com/nicmsri/cosmogp.git
cd cosmogp
pip install -r requirements.txt
```
### Usage

Find a basic example to get started with `cosmogp` in the `tutorial.ipynb` notebook.

### Acknowledgments

This package was developed as part of my master thesis work. It is primarily intended for research purposes rather than general coding use. For more details on the theoretical background and the specific applications, please refer to [my master thesis](https://github.com/nicmsri/mysliceofweb/blob/main/Weak_lensing_map_inference__a_physics_informed_Gaussian_processes_approach.pdf).
