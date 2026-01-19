import os
import pickle
import torch
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
import wandb
from pymatgen.core import Structure


STRUCT_PKL = "processed_structures_CHGNet/train_structures.pkl"
OUT_DIR = "finetuneOut"
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
    # 1. LOAD DATA
    structures = load_structures()
    energies = get_energies(structures)
    
    dataset = StructureData(
        structures=structures, 
        energies=energies, 
        forces=None, stresses=None, magmoms=None
    )

    # 2. DATA LOADERS
    # Note: accessing config as a dict since we aren't using wandb.config here
    train_loader, val_loader, _ = get_train_val_test_loader(
        dataset, 
        batch_size=config['batch_size'], 
        train_ratio=0.95, 
        val_ratio=0.05
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CHGNet.load().to(device)
    
    if config['freeze_embeddings']:
        for layer in [model.atom_embedding, model.bond_embedding, model.angle_embedding]:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 3. INITIALIZE TRAINER (This handles W&B)
    # Generate a unique run name for this config
    run_name = f"bs{config['batch_size']}_lr{config['learning_rate']}_{config['optimizer']}"
    
    trainer = Trainer(
        model=model,
        targets="e",
        optimizer=config['optimizer'],
        scheduler=config['scheduler'],
        criterion="MSE",
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        use_device=device,
        print_freq=10,
        wandb_path=f"CHGNet-finetuning/{run_name}" 
    )
    
    # 4. TRAIN
    trainer.train(train_loader, val_loader, None)
    
    # 5. SAVE BEST MODEL
    best_model = trainer.best_model or trainer.model
    model_path = os.path.join(OUT_DIR, f"model_{run_name}.pt")
    torch.save(best_model.state_dict(), model_path)


if __name__ == "__main__":
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Define hyperparameter grid manually
    configs = [
        {"batch_size": 8, "learning_rate": 1e-3, "epochs": 50, "freeze_embeddings": False, "optimizer": "Adam", "scheduler": "CosLR"},
        {"batch_size": 16, "learning_rate": 5e-4, "epochs": 50, "freeze_embeddings": True, "optimizer": "AdamW", "scheduler": "StepLR"},
        {"batch_size": 32, "learning_rate": 1e-4, "epochs": 50, "freeze_embeddings": False, "optimizer": "Adam", "scheduler": "CosLR"},
    ]
    
    for config in configs:
        print(f"\n{'='*50}\nTraining with config: {config}\n{'='*50}")
        train_with_config(config)


# if __name__ == "__main__":
#     # ensure W&B uses WANDB_API_KEY if set; relogin forces refresh of credentials
#     wandb.login(key=os.environ.get("WANDB_API_KEY"), relogin=True)

#     # quick check to show which user/entity the API key belongs to
#     try:
#         viewer = wandb.Api().viewer  # property, not a method
#         print("W&B viewer:", viewer)
#         username = getattr(viewer, "username", None)
#         if username:
#             print("W&B username:", username)
#     except Exception as e:
#         print("W&B authentication/permission error:", type(e).__name__, e)
#         raise

#     # Create and run sweep
#     try:
#         sweep_id = wandb.sweep(
#             sweep_config,
#             project="chgnet-2d-materials"
#         )
#         print(f"Sweep ID: {sweep_id}")
#         wandb.agent(sweep_id, function=train_with_config, count=20)
#     except Exception as e:
#         print("W&B sweep error:", type(e).__name__, e)
#         raise
