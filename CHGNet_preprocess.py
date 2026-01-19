from pymatgen.core import Structure, Lattice
from ase import Atoms
import pandas as pd
import numpy as np
import os
import pickle

data = pd.read_json('train.json')
all_structures = []
errors = 0
error_indices = []

def fix_atoms_dict(atoms_dict):
    """Convert lists back to numpy arrays for ASE compatibility."""
    fixed = atoms_dict.copy()
    # Fields that ASE expects as numpy arrays
    array_fields = ['numbers', 'positions', 'cell', 'masses', 'momenta', 
                    'initial_charges', 'initial_magmoms', 'tags']
    
    for field in array_fields:
        if field in fixed and fixed[field] is not None:
            fixed[field] = np.array(fixed[field])
    
    # Handle pbc separately (can be bool or array of bools)
    if 'pbc' in fixed:
        fixed['pbc'] = np.array(fixed['pbc'])
    
    return fixed


if __name__ == "__main__":
    for idx, row in data.iterrows():
        try:
            # Fix dict to avoid assertion errors
            atoms_dict = fix_atoms_dict(row['atoms'])
            ase_obj = Atoms.fromdict(atoms_dict)
            
            lattice = np.array(ase_obj.get_cell())
            positions = ase_obj.get_positions()
            atomic_numbers = ase_obj.get_atomic_numbers()
            
            # Makes a 3D vacuum lattice if 2D material represented
            c_length = np.linalg.norm(lattice[2]) # c is the 3rd lattice vector
            if c_length < 0.1:
                lattice[2] = [0, 0, 20.0] # Makes it 20 Angstroms long in z-direction
            
            if np.abs(np.linalg.det(lattice)) < 1e-6:
                errors += 1
                error_indices.append(idx)
                continue
            
            # Create Structure using atomic numbers
            struct = Structure(
                lattice=Lattice(lattice),
                species=atomic_numbers.tolist(),
                coords=positions,
                coords_are_cartesian=True,
                to_unit_cell=True
            )
            
            struct.properties = {
                "id": row['id'],
                "hform": row['hform'], # Comment in case your are processing the test data
                "data_og_formula": row['formula']
            }
            
            all_structures.append(struct)
            
        except Exception as e:
            print(f"Error at index {idx}: {type(e).__name__} - {e}")
            if len(error_indices) < 5:
                print(f"  atoms_dict keys: {row['atoms'].keys()}")
            errors += 1
            error_indices.append(idx)

    print(f"\nSuccess! Converted {len(all_structures)} structures.")
    print(f"Failed: {errors} structures")

    # Save processed structures
    out_dir = "processed_structures_CHGNet"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_structures.pkl"), "wb") as f:
        pickle.dump(all_structures, f)