from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import os


DIR = "C:\\Users\\jacop\\Desktop\\DTU\\Material Design with ML\\10316-materials-design-with-ml-and-ai-jan-26"


test = pd.read_json(os.path.join(DIR, "test.json"))
train = pd.read_json(os.path.join(DIR, 'train.json'))

# Assuming 'data' is your dictionary/row
ase_obj = train['atoms']

# Convert ASE Atoms to Pymatgen Structure
structure = AseAtomsAdaptor.get_structure(ase_obj)

# You can then attach your other data as site or crystal properties if needed
structure.properties = {
    "material_id": train['id'],
    "hform": train['hform']
}

print(structure)