import os
import pickle
import torch
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
import wandb
from pymatgen.core import Structure


STRUCT_PKL = "processed_structures_CHGNet/train_structures.pkl"
OUT_DIR = "finetune_out"
os.makedirs(OUT_DIR, exist_ok=True)

# Hyperparameter sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {'name': 'val_e_MAE', 'goal': 'minimize'},
    'parameters': {
        'batch_size': {'values': [8, 16, 32]},
        'learning_rate': {'min': 1e-4, 'max': 1e-2, 'distribution': 'log_uniform'},
        'epochs': {'value': 50},
        'freeze_embeddings': {'values': [True, False]},
        'optimizer': {'values': ['Adam', 'AdamW']},
        'scheduler': {'values': ['CosLR', 'StepLR']},
    }
}

# 1) Load processed structures (with validation)
def load_structures():
    
    with open(STRUCT_PKL, "rb") as f:
        raw_structs = pickle.load(f)
    
    structures = []
    failed = []
    for i, s in enumerate(raw_structs):
        try:
            if isinstance(s, dict):
                s = Structure.from_dict(s)
            if not isinstance(s, Structure):
                raise TypeError(f"Unsupported type: {type(s)}")
            _ = s.composition  # validate
            structures.append(s)
        except Exception as e:
            failed.append((i, type(e).__name__))
    
    if failed:
        print(f"Skipped {len(failed)} invalid structures")
    
    return structures

def get_energies(structures):
    energies = []
    for s in structures:
        e = None
        if hasattr(s, "properties") and isinstance(s.properties, dict):
            e = s.properties.get("hform") or s.properties.get("energy_per_atom")
        energies.append(float(e) if e is not None else None)
    return energies

def train_with_config(config=None):
    with wandb.init(config=config, project="chgnet-2d-materials"):
        config = wandb.config
        
        # Load data
        structures = load_structures()
        energies = get_energies(structures)
        
        # Build dataset
        dataset = StructureData(
            structures=structures, 
            energies=energies, 
            forces=None, 
            stresses=None, 
            magmoms=None
        )

        # Make only training and validation sets
        train_loader, val_loader, _ = get_train_val_test_loader(
            dataset, 
            batch_size=config.batch_size, 
            train_ratio=0.95, 
            val_ratio=0.05
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CHGNet.load().to(device) # Load pre-trained CHGNet
        
        # Optionally freeze embeddings
        if config.freeze_embeddings:
            for layer in [model.atom_embedding, model.bond_embedding, model.angle_embedding]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Create trainer
        trainer = Trainer(
            model=model,
            targets="e",
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            criterion="MSE",
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            use_device=device,
            print_freq=10,
            wandb_path=OUT_DIR,
        )
        
        trainer.train(train_loader, val_loader, None)
        
        # Save best model
        best_model = trainer.best_model or trainer.model
        model_path = os.path.join(OUT_DIR, f"model_{wandb.run.name}.pt")
        torch.save(best_model.state_dict(), model_path)
        
        # Log final metrics
        wandb.log({
            "final_val_e_MAE": trainer.best_val_score,
            "model_path": model_path
        })

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="chgnet-2d-materials")
    wandb.agent(sweep_id, function=train_with_config, count=20)