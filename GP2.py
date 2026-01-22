# %%
import jax
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from scipy.sparse import csr_matrix, vstack
import hashlib
import jax.numpy as jnp
from multiprocessing.pool import ThreadPool
from ase import Atoms
import json
from tqdm import tqdm
from dscribe.descriptors import MBTR

jax.config.update("jax_enable_x64", True)
#%%
# Load training data
train = pd.read_json(r"data\train.json")
# Transform atoms entry to ASE atoms object
train['atoms'] = train['atoms'].apply(lambda x: Atoms(**x))
print(train.shape)

#%%
# Compute max number of atoms for padding
number_of_atoms = []
for atom in train['atoms']:
    number_of_atoms.append(len(atom))

max_number_of_atoms = np.max(number_of_atoms)
print(f'Max {max_number_of_atoms} atoms')

#%%
# Define Coulomb matrix fingerprint function with padding
def coulomb_matrix_padding(atoms, n_atoms_max):
    # Creating a matrix with the product of the atomic numbers such that M_ij = Z[i]*Z[j]
    atomic_numbers = np.outer(atoms.get_atomic_numbers(),atoms.get_atomic_numbers())
    # Getting the distance matrix of the atoms object, such that element D_ij = |r_i - r_j|
    distances = atoms.get_all_distances()
    # Setting the diagonal elements, which are all zero, to 1 to avoid overflow errors
    np.fill_diagonal(distances, 1)
    
    # Creating the Cmat
    cmat = np.multiply(atomic_numbers, 1/distances) # Element wise multiplication
    np.fill_diagonal(cmat, 0.5*np.array(atoms.get_atomic_numbers())**2.4) # Filling the diagonal as described in the slides
    
    
    # Padding the matrix with zeros such that all of the fingerprints have the same size
    shape = cmat.shape
    if shape[0] < n_atoms_max:
        # Padding the matrix
        pad_h = np.zeros((n_atoms_max - shape[0], shape[0]))
        pad_v = np.zeros((n_atoms_max, n_atoms_max - shape[0]))
        cmat = np.vstack((cmat, pad_h))
        cmat = np.hstack((cmat, pad_v))
    
    # Sorting according to l2 norm of columns
    sortidxs = np.flip(np.linalg.norm(cmat, axis=0).argsort())
    cmat = cmat[sortidxs][:,sortidxs]
    
    return cmat
#%%
# Generate fingerprints for the whole dataset
print("Generating fingerprints...")
cmats = np.zeros((len(train), max_number_of_atoms**2))
for i, atoms in enumerate(train['atoms']):
    if i % 1000 == 0:
        print(f"Processing structure {i}")
    cmat = coulomb_matrix_padding(atoms, max_number_of_atoms)
    cmats[i, :] = cmat.flatten()

print(f"Fingerprints shape: {cmats.shape}")
#%%
# Set up features and targets
X = pd.DataFrame(data=cmats, index=train.index)
y = train['hform'].values

print(f'X: {X.shape}')
print(f'y: {y.shape}')

#%%
# Split data into training and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=250)

#%%
# Apply PCA for dimensionality reduction
n_comp_PCA = 10
pca = PCA(n_components=n_comp_PCA)
scaler_x = StandardScaler()
Xtrain = pca.fit_transform(scaler_x.fit_transform(Xtrain))
Xtest = pca.transform(scaler_x.transform(Xtest))

scaler_y = StandardScaler()
ytrain_scaled = scaler_y.fit_transform(ytrain.reshape(-1, 1))
ytest_reshaped = ytest.reshape(-1, 1) # Keep raw for final comparison

print(f"With {n_comp_PCA} PCA components {100*np.sum(pca.explained_variance_ratio_):0.2f}% of the variance is explained")
print(f"X_train: {Xtrain.shape}")
print(f"X_test: {Xtest.shape}")

#%%
print(f"ytrain stats - min: {ytrain.min()}, max: {ytrain.max()}, mean: {ytrain.mean()}")
print(f"ytest stats - min: {ytest.min()}, max: {ytest.max()}, mean: {ytest.mean()}")
ytrain_reshaped = ytrain.reshape(-1, 1)
ytest_reshaped = ytest.reshape(-1, 1)

# %%

class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel, kappa=1., lengthscale=1., sigma=1/2, jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
            kappa            -- magnitude (positive scalar)
            lengthscale      -- characteristic lengthscale (positive scalar)
            sigma            -- noise std. dev. (positive scalar)
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(kappa, lengthscale, sigma)
        self.check_dimensions()

    def check_dimensions(self):
        assert self.X.ndim == 2, f"The variable X must be of shape (N, D), however, the current shape is: {self.X.shape}"
        N, D = self.X.shape

        assert self.y.ndim == 2, f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        assert self.y.shape == (N, 1), f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        

    def set_hyperparameters(self, kappa, lengthscale, sigma):
        self.kappa = kappa
        self.lengthscale = lengthscale
        self.sigma = sigma

    def posterior_samples(self, key, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                key              -- jax random key for controlling the random number generator
                Xstar            -- PxD prediction points
        
            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        mu, Sigma = self.predict_f(Xstar)
        f_samples = generate_samples(key, mu.ravel(), Sigma, num_samples)
        
        ##############################################
        # End of solution
        ##############################################

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples
        
    def predict_y(self, Xstar):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare relevant matrices
        mu, Sigma = self.predict_f(Xstar)
        Sigma = Sigma + self.sigma**2 * jnp.identity(len(mu))
        
        ##############################################
        # End of solution
        ##############################################

        return mu, Sigma

    def predict_f(self, Xstar):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare relevant matrices
        k = self.kernel.contruct_kernel(Xstar, self.X, self.kappa, self.lengthscale, jitter=self.jitter)
        K = self.kernel.contruct_kernel(self.X, self.X, self.kappa, self.lengthscale, jitter=self.jitter)
        Kstar = self.kernel.contruct_kernel(Xstar, Xstar, self.kappa, self.lengthscale, jitter=self.jitter)
        
        # Compute C matrix
        C = K + self.sigma**2*jnp.identity(len(self.X)) 

        # computer mean and Sigma
        mu = jnp.dot(k, jnp.linalg.solve(C, self.y))
        Sigma = Kstar - jnp.dot(k, jnp.linalg.solve(C, k.T))
        
        ##############################################
        # End of solution
        ##############################################

        # sanity check for dimensions
        assert (mu.shape == (len(Xstar), 1)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, 1), but actual shape was {mu.shape}. Please check implementation"
        assert (Sigma.shape == (len(Xstar), len(Xstar))), f"The shape of the posterior Sigma seems wrong. Expected ({len(Xstar)}, {len(Xstar)}), but actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma
    
    def log_marginal_likelihood(self, kappa, lengthscale, sigma):
        """ 
            evaluate the log marginal likelihood p(y) given the hyperparaemters 

            Arguments:
            kappa       -- positive scalar 
            lengthscale -- positive scalar
            sigma       -- positive scalar
            """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare kernels
        K = self.kernel.contruct_kernel(self.X, self.X, kappa, lengthscale)
        C = K + sigma**2*jnp.identity(self.N)

        # compute Cholesky decomposition
        L = jnp.linalg.cholesky(C)
        v = jnp.linalg.solve(L, self.y)

        # compute log marginal likelihood
        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        quad_term =  0.5*jnp.sum(v**2) 
        const_term = -0.5*self.N*jnp.log(2*jnp.pi)

        return const_term - logdet_term - quad_term
        
        ##############################################
        # End of solution
        ##############################################

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa=1., lengthscale=1.0):
        """
            the argument kernel_fun must be a function of three arguments kernel_fun(||tau||, kappa, lengthscale), e.g. 
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*np.exp(-0.5*tau**2/lengthscale**2)
        """
        self.kernel_fun = kernel_fun
        self.kappa = kappa
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa=None, lengthscale=None, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            kappa           -- magnitude (positive scalar)
            lengthscale     -- characteristic lengthscale (positive scalar)
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # prep hyperparameters
        kappa = self.kappa if kappa is None else kappa
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        ##############################################
        # Your solution goes here
        ##############################################
        
        # compute all the pairwise distances efficiently
        dists = jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0))**2, axis=-1))
        
        # squared exponential covariance function
        K = self.kernel_fun(dists, kappa, lengthscale)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter*jnp.identity(len(X1))
        
        ##############################################
        # End of solution
        ##############################################
        
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K
    

def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*jnp.exp(-0.5*tau**2/lengthscale**2)


# %%
kernel = StationaryIsotropicKernel(squared_exponential)
gp_post = GaussianProcessRegression(jnp.array(Xtrain), jnp.array(ytrain_scaled),  
                                     kernel, kappa=0.602, lengthscale=3.24, sigma=0.605,jitter=0)
mu_scaled, Sigma_scaled = gp_post.predict_y(jnp.array(Xtest))
preds_eV = scaler_y.inverse_transform(mu_scaled)

# %%
# Optimize hyperparameters (kappa, lengthscale, sigma) for the GP class using scipy.optimize.minimize
def neg_log_marginal_likelihood(params):
    kappa, lengthscale, sigma = params
    return -gp_post.log_marginal_likelihood(kappa, lengthscale, sigma)

res = sp.optimize.minimize(
    neg_log_marginal_likelihood,
    [1, 10, 0.01],  
    method='L-BFGS-B',
    bounds=sp.optimize.Bounds(
    [0.1, 0.1, 1e-4], [10.0, 50.0, 1.0]),)

opt_kappa, opt_lengthscale, opt_sigma = res.x
print(f"Optimal kappa: {opt_kappa}, Optimal lengthscale: {opt_lengthscale}, Optimal sigma: {opt_sigma}")

# Refit GP with optimal hyperparameters and predict
gp_post_opt = GaussianProcessRegression(
    jnp.array(Xtrain), jnp.array(ytrain_scaled),  # <-- use ytrain_scaled
    kernel, kappa=opt_kappa, lengthscale=opt_lengthscale, sigma=opt_sigma,
    jitter=0
)
mu_scaled, Sigma_scaled = gp_post_opt.predict_y(jnp.array(Xtest))

preds_eV = scaler_y.inverse_transform(mu_scaled)

#%%
# plots and rmse
plt.scatter(ytest, preds_eV[:,0])
plt.xlim([ytest.min()-0.5, ytest.max()+0.5])
plt.ylim([ytest.min()-0.5, ytest.max()+0.5])
plt.xlabel("True values")
plt.ylabel("Predicted values")
# plt.errorbar(ytest, preds[0][:,0], yerr= np.sqrt(np.diag(preds[1])), fmt='o')
rmse = np.sqrt(np.mean((ytest - preds_eV[:,0]) ** 2))
print("RMSE:", rmse)
# %%
# do a grid search over hyperparameters
from tqdm import tqdm
from scipy import sparse as sp
kappa_list = [1.0, 2.0, 5]
lengthscale_list = [1.5, 2.5, 5]
sigma_list = [0.2, 0.4, 0.6,0.8]

results = []
total = len(kappa_list) * len(lengthscale_list) * len(sigma_list)
for kappa in tqdm(kappa_list, desc="Kappa"):
    for lengthscale in tqdm(lengthscale_list, desc="Lengthscale", leave=False):
        for sigma in tqdm(sigma_list, desc="Sigma", leave=False):
            gp = GaussianProcessRegression(
                jnp.array(Xtrain), jnp.array(ytrain_scaled),
                kernel, kappa=kappa, lengthscale=lengthscale, sigma=sigma
            )
            mu_scaled, Sigma_scaled = gp.predict_y(jnp.array(Xtest))
            preds_eV = scaler_y.inverse_transform(mu_scaled)
            rmse = np.sqrt(np.mean((ytest - preds_eV[:,0]) ** 2))
            results.append({
                'kappa': kappa,
                'lengthscale': lengthscale,
                'sigma': sigma,
                'rmse': rmse
            })
            print(f'kappa: {kappa}, lengthscale: {lengthscale}, sigma: {sigma}, rmse: {rmse}')
# %%
to_submit = pd.read_json(r"data\test.json")
to_submit['atoms'] = to_submit['atoms'].apply(lambda x: Atoms(**x))

x_to_predict = np.zeros((len(to_submit), max_number_of_atoms**2))
for i, atoms in enumerate(to_submit['atoms']):
    cmat = coulomb_matrix_padding(atoms, max_number_of_atoms)
    x_to_predict[i, :] = cmat.flatten()
X_submit = pd.DataFrame(data=x_to_predict, index=to_submit.index)
X_submit_pca = pca.transform(scaler_x.transform(X_submit))
mu_submit_scaled, Sigma_submit_scaled = gp_post_opt.predict_y(jnp.array(X_submit_pca))
preds_submit_eV = scaler_y.inverse_transform(mu_submit_scaled)

# Create submission CSV in the form of id,hform
submission = pd.DataFrame({
    'id': to_submit['id'],
    'hform': preds_submit_eV[:, 0]
})
submission.to_csv('submission.csv', index=False)
# %%
# Optimal kappa: 0.5608138007621044, Optimal lengthscale: 1.8435787933970478, Optimal sigma: 0.8522635708851426
