# %%
import numpy as np
import ase.db
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split

# # %%
# dbase = ase.db.connect("../cubic_perovskites.db")
# # Do not include the so-called "reference systems"
# syss = [c for c in dbase.select() if not hasattr(c, "reference")]

# Aset = set()
# Bset = set()
# anionset = set()
# for p in syss:
#     Aset.add(p.A_ion)
#     Bset.add(p.B_ion)
#     anionset.add(p.anion)
# Alist = list(Aset)
# Alist.sort()
# Blist = list(Bset)
# Blist.sort()
# anionlist = list(anionset)
# anionlist.sort()

# aniondict = {
#     "N3": [0, 3, 0, 0],
#     "O2F": [2, 0, 0, 1],
#     "O2N": [2, 1, 0, 0],
#     "O2S": [2, 0, 1, 0],
#     "O3": [3, 0, 0, 0],
#     "OFN": [1, 1, 0, 1],
#     "ON2": [1, 2, 0, 0],
# }

# elemdict = {
#     "Ag": [5, 11],
#     "Al": [3, 13],
#     "As": [4, 15],
#     "Au": [6, 11],
#     "B": [2, 13],
#     "Ba": [6, 2],
#     "Be": [2, 2],
#     "Bi": [6, 15],
#     "Ca": [4, 2],
#     "Cd": [5, 12],
#     "Co": [4, 9],
#     "Cr": [4, 6],
#     "Cs": [6, 1],
#     "Cu": [4, 11],
#     "Fe": [4, 8],
#     "Ga": [4, 13],
#     "Ge": [4, 14],
#     "Hf": [6, 4],
#     "Hg": [6, 12],
#     "In": [5, 13],
#     "Ir": [6, 9],
#     "K": [4, 1],
#     "La": [6, 2.5],
#     "Li": [2, 1],
#     "Mg": [3, 2],
#     "Mn": [4, 7],
#     "Mo": [5, 6],
#     "Na": [3, 1],
#     "Nb": [5, 5],
#     "Ni": [4, 10],
#     "Os": [6, 8],
#     "Pb": [6, 14],
#     "Pd": [5, 10],
#     "Pt": [6, 10],
#     "Rb": [5, 1],
#     "Re": [6, 7],
#     "Rh": [5, 9],
#     "Ru": [5, 8],
#     "Sb": [5, 15],
#     "Sc": [4, 3],
#     "Si": [3, 14],
#     "Sn": [5, 14],
#     "Sr": [5, 2],
#     "Ta": [6, 5],
#     "Te": [5, 16],
#     "Ti": [4, 4],
#     "Tl": [6, 13],
#     "V": [4, 5],
#     "W": [6, 6],
#     "Y": [5, 3],
#     "Zn": [4, 12],
#     "Zr": [5, 4],
# }
# # %%
# print(syss)
# def fingerprint(row):
#     A = elemdict[row.A_ion]
#     B = elemdict[row.B_ion]
#     anion = aniondict[row.anion]
#     features = A + B + anion
#     return features


# X = np.array([fingerprint(row) for row in syss], dtype=np.int16)
# y = np.array([row.heat_of_formation_all for row in syss], dtype=np.float32)


# np.save("X.npy", X)
# np.save("y.npy", y)

# %%

X = np.load("X.npy")
y = np.load("y.npy")

# Only use 500
mask = np.random.choice(X.shape[0], 2000, replace=False)
X500 = X[mask]
y500 = y[mask]



def kernel(x1, x2, ell=1.0, k0=1.0):
    diff = x1 - x2
    return k0 * np.exp(-0.5 * np.dot(diff, diff) / ell**2)

def kernel_matrix(X, ell=1.0, k0=1.0):
    # X should be a 2D array of shape (n_samples, n_features)
    X1 = X[:, np.newaxis, :]  # shape: (n_samples, 1, n_features)
    X2 = X[np.newaxis, :, :]  # shape: (1, n_samples, n_features)
    diff = np.array(X1 - X2, dtype=np.int16)  # Broadcasting to shape: (n_samples, n_samples, n_features)
    sqdist = np.sum(diff**2, axis=-1)  # shape: (n_samples, n_samples)
    return k0 * np.exp(-0.5 * sqdist / ell**2)

# K = kernel_matrix(X, ell=1.0)
# %%
# np.save("K.npy", K)

# %%
# ell, k0, sigma 
sigma = 0.1
N = X500.shape[0]

def logP(theta,sigma):
    K = kernel_matrix(X500, ell=theta[0], k0=theta[1])
    C = K + sigma**2*np.identity(N)

    # Use Cholesky decomposition for numerical stability
    try:
        L = np.linalg.cholesky(C)
        # Solve using Cholesky
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y500))
        eigvals, eigvecs = np.linalg.eigh(C)
        log_det_C = np.sum(np.log(eigvals))
        return -0.5 * log_det_C - 0.5 * np.dot(y500, alpha)
    except np.linalg.LinAlgError:
        # Return a large negative value if matrix is not positive definite
        return -1e10


result = sp.optimize.minimize(lambda theta: -logP(theta, sigma=sigma), 
                            [1,1], bounds=sp.optimize.Bounds([1e-3,1e-3],[10,10]),
                            options={
                            'ftol': 1e-6})

# %%
print(result)
# %%

ell, k0 = result.x

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X500, y500, test_size=0.2)

K = kernel_matrix(Xtrain, ell=ell, k0=k0)
C = K + sigma**2 * np.identity(Xtrain.shape[0])


C_inv = np.linalg.inv(C)
pred = np.zeros(Xtest.shape[0])
sigmas = np.zeros(Xtest.shape[0])
for i, x in enumerate(Xtest):
    k_s = np.array([kernel(x, xt, ell=ell, k0=k0) for xt in Xtrain])
    y_pred = k_s.T @ C_inv @ (ytrain)
    pred[i] = y_pred

    s = kernel(x, x, ell=ell, k0=k0) - k_s.T @ C_inv @ k_s
    sigmas[i] = s



mse = np.mean((ytest - pred) ** 2)
print("MSE:", mse)

# %%

# Plot predictions with uncertainty
plt.scatter(ytest, pred)
plt.errorbar(ytest, pred, yerr= np.sqrt(sigmas), fmt='o')

# %%
diffs = ytest - pred
plt.hist(diffs/np.sqrt(sigmas), bins=20, density=True)

mean_diff = np.mean(diffs/np.sqrt(sigmas))
var_diff = np.var(diffs/np.sqrt(sigmas))

plt.plot(np.linspace(-4,4,100), sp.stats.norm.pdf(np.linspace(-4,4,100), loc=mean_diff, scale=var_diff), label='Normal Distribution')
plt.title(f"Mean: {mean_diff:.2f}, Variance: {var_diff:.2f}")
# %%
print(np.mean(sigmas))

# %%
