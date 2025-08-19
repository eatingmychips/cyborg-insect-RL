import os
import sys
import torch
import logging
from pathlib import Path
import glob

# Add your project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.policy_net import PolicyNetwork  
from behavioral_cloning import BCTrainer, DemonstrationDataLoader
from behavioral_cloning.evaluate_bc import plot_training_history


####### Define csv file path for training ########
csv_file_path = r"G:\biorobotics\data\ClosedLoopControl\MiscDataCollection\BC_Folder"
csv_files = glob.glob(os.path.join(csv_file_path, "*.csv"))
csv_files = sorted(csv_files)  # Sort for consistent ordering


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Configuration
    config = {
        'data': {
            'csv_paths': csv_files,
            'state_columns': ['heading_error', 'path_distance', 'progress'],  # Your reduced state
            'action_column': 'action',
            'batch_size': 32,
            'train_split': 0.8,
            'downsample': True,
            'keep_ratio': 0.05, 
            'random_seed': 42
        },
        'model': {
            'state_dim': 3,  # Based on your reduced state space
            'action_dim': 13,  # Number of discrete actions (adjust as needed)
            'hidden_dims': [128, 64]  # Network architecture
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-3,
            'patience': 15,
            'save_path': 'data/checkpoints/bc_policy.pt'
        }
    }
    
    # Create directories
    os.makedirs('data/checkpoints', exist_ok=True)
    os.makedirs('results/bc_plots', exist_ok=True)
    
    # Initialize data loader
    data_loader = DemonstrationDataLoader(
        state_columns=config['data']['state_columns'],
        action_column=config['data']['action_column'],
        normalize=True,
        train_split=config['data']['train_split'],
        downsample=config['data']['downsample'],
        keep_ratio=config['data']['keep_ratio'],
        random_seed=config['data']['random_seed']

    )
    
    # Create dataloaders
    train_loader, val_loader = data_loader.create_dataloaders(
        csv_paths=config['data']['csv_paths'],
        batch_size=config['data']['batch_size']
    )
    
    # Save scaler for RL consistency
    data_loader.save_scaler('data/checkpoints/state_scaler.pkl')
    logging.info("Saved state scaler for RL consistency")
    
    # Initialize policy network
    policy = PolicyNetwork(
        input_dim=config['model']['state_dim'],
        output_dim=config['model']['action_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    # Initialize trainer
    trainer = BCTrainer(
        policy_network=policy,
        learning_rate=config['training']['learning_rate']
    )
    
    # Train the policy
    logging.info("Starting behavioral cloning training...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        save_path=config['training']['save_path']
    )
    
    # Plot training history
    plot_training_history(
        train_losses=training_results['train_losses'],
        val_losses=training_results['val_losses'],
        val_accuracies=training_results['val_accuracies'],
        save_path='results/bc_plots/training_history.png'
    )
    
    logging.info(f"Training completed! Best validation loss: {training_results['best_val_loss']:.4f}")
    logging.info(f"Model saved to: {config['training']['save_path']}")

if __name__ == "__main__":
    main()
