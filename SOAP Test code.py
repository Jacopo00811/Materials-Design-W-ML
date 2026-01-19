import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import json

# MOUNT DRIVE
from google.colab import drive
drive.mount('/content/drive')

# ---------------------------------------------------------
# 1. SETUP DATA (THE FIX)
# ---------------------------------------------------------
print("1. Loading data from Drive...")

# FIX: Use pd.read_json directly with the Drive path. 
# Do not use json.load() here.
train_df = pd.read_json('/content/drive/MyDrive/Json/train.json')
test_df = pd.read_json('/content/drive/MyDrive/Json/test.json')

# Convert text to Atoms objects
train_df['atoms_object'] = train_df['atoms'].apply(lambda x: Atoms(**x))
test_df['atoms_object'] = test_df['atoms'].apply(lambda x: Atoms(**x))

# ---------------------------------------------------------
# 2. FIND ALL ATOMIC SPECIES
# ---------------------------------------------------------
print("Finding all unique elements in the dataset...")
all_atoms_list = train_df['atoms_object'].tolist() + test_df['atoms_object'].tolist()
unique_species = set()
for mol in all_atoms_list:
    unique_species.update(mol.get_atomic_numbers())

species_list = sorted(list(unique_species))
print(f"Found {len(species_list)} unique elements: {species_list}")

# ---------------------------------------------------------
# 3. SETUP SOAP (LITE MODE FOR COLAB)
# ---------------------------------------------------------
print("2. Setting up SOAP Descriptor...")

# I strongly recommend keeping these optimizations enabled.
# If you run out of RAM, decrease n_max or l_max slightly.
soap = SOAP(
    species=species_list,
    periodic=True,
    r_cut=6.0,
    n_max=3,             # Good balance for Colab
    l_max=2,             # Good balance for Colab
    average="inner",
    sparse=False,
    dtype="float32",                 # <--- Saves 50% RAM
    compression={"mode": "crossover"} # <--- Critical to prevent crashing
)

print(f"Features per material: {soap.get_number_of_features()}")

# Create fingerprints
print("Calculating SOAP fingerprints...")
# Combine lists to calculate all at once
all_atoms = train_df['atoms_object'].tolist() + test_df['atoms_object'].tolist()
X_all = soap.create(all_atoms)

# Split back into train and test
X_train_raw = X_all[:len(train_df)]
X_submission_raw = X_all[len(train_df):]
y_train_raw = train_df['hform'].values

# 4. VERIFY
print(f"Shape of X_train: {X_train_raw.shape}")

# ---------------------------------------------------------
# 4. PREPARE FOR AI
# ---------------------------------------------------------
scaler = StandardScaler() # Scaling forces all numbers to roughly the same range (centered at 0), so the model treats every feature fairly.
X_train_scaled = scaler.fit_transform(X_train_raw)
X_submission_scaled = scaler.transform(X_submission_raw)

# Split training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train_raw, test_size=0.2, random_state=42 # test size is 20 %.
)

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------
print("3. Training Gaussian Process...")

# Colab is faster than a laptop, but GPs are still slow.
# Try 2000 samples first. If it finishes fast, increase to 4000 or 5000.
subset_size = 3000
X_train_small = X_train[:subset_size]
y_train_small = y_train[:subset_size]

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

model.fit(X_train_small, y_train_small)

# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
print("4. Evaluating...")
score = model.score(X_val, y_val)
print(f"Accuracy (R^2) on validation set: {score:.3f}")

# Predict
predictions = model.predict(X_submission_scaled)

submission = pd.DataFrame({'id': test_df['id'], 'hform': predictions})
submission.to_csv('submission.csv', index=False)
print("Done! Saved to submission.csv")

print("--- VERIFICATION ---")
# 1. Check Dimensions
print(f"Number of Materials: {X_train.shape[0]}") # Should be 8000 (or length of train_df)
print(f"Length of Fingerprint: {X_train.shape[1]}") # Should be ~400 (CM) or ~1000 (SOAP)

# 2. Check Data Integrity
# Are there any NaNs (Not a Number) or Infinities? (This ruins Gaussian Models)
if np.isnan(X_train).any() or np.isinf(X_train).any():
    print("CRITICAL ERROR: Your fingerprints contain NaNs or Infinity!")
else:
    print("SUCCESS: Fingerprints are clean numbers.")

# 3. Visual Check
print("Example Fingerprint (First 10 numbers):")
print(X_train[0][:10])



import matplotlib.pyplot as plt

# 1. Get predictions for the Validation set (the "Test" you kept secret)
y_pred_val = model.predict(X_val)

# 2. Create the Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_val, y_pred_val, alpha=0.5, s=10, color='blue', label='Predictions')

# 3. Draw the Perfect Line (x = y)
# If a dot is on this line, the prediction was 100% correct
min_val = min(y_val.min(), y_pred_val.min())
max_val = max(y_val.max(), y_pred_val.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

# 4. Labels
plt.xlabel('Actual Heat of Formation (eV)')
plt.ylabel('Predicted Heat of Formation (eV)')
plt.title(f'Model Accuracy Check\nR^2 Score: {score:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()