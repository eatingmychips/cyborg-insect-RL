import torch
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_bc_policy(policy: torch.nn.Module, data_loader, 
                      action_names: List[str] = None, 
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
    """Comprehensive evaluation of behavioral cloning policy."""
    
    policy.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for states, actions in data_loader:
            states = states.to(device)
            
            # Get predictions
            outputs = policy(states)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(actions.numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_targets)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    if action_names:
        class_report = classification_report(all_targets, all_predictions, 
                                           target_names=action_names, 
                                           output_dict=True)
    else:
        class_report = classification_report(all_targets, all_predictions, 
                                           output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         val_accuracies: List[float], save_path: str = None):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Accuracy plot
    ax2.plot(val_accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, action_names: List[str] = None, 
                         save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=action_names, yticklabels=action_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_policy_in_env(policy: torch.nn.Module, env, scaler=None, 
                      num_episodes: int = 5, max_steps: int = 1000,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """Test the behavioral cloning policy in the actual environment."""
    
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps):
            # Preprocess state (apply same normalization as training)
            if scaler:
                state_normalized = scaler.transform(state.reshape(1, -1))
            else:
                state_normalized = state.reshape(1, -1)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_normalized).to(device)
            
            # Get action
            with torch.no_grad():
                action_probs = policy(state_tensor)
                action = torch.argmax(action_probs, dim=1).cpu().numpy()[0]
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards
    }
