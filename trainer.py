
"""
Create a trainer that optimzes the model and allows for relevant properties to be shown
"""
import pdb
import torch
from torch.utils.data import Subset, DataLoader
import torch.nn as nn


import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from scoring_model import BaselineScorer
from dataset import Preprocessor, LGEDataset
class TrainingConfig():
    def __init__(self): 
        self.learning_rate = 0.0001
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.num_epochs = 5
class Trainer():
    """
    Class for Training LGE Model 


    Requirements: / TODO

    - init - TrainingConfig Object will tell how the optimization will behave. 
    - after every epoch trainer will call evaluator to calculate all relevant statistics
    - valildatiotn flow 
    - forward + backward optitmizer

    """

    def __init__(self, model, train_loader,evaluator,  val_loader, config): 

        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.evaluator = evaluator

        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = config.learning_rate
        )
        pos_weight = torch.tensor([2.0]).to(config.DEVICE) # Error I need to push this into the devicd
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    def train_epoch(self):
        """Single training epoch."""
        #TODO: Evaluate to see if this correct and ideal for my kind of data
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target, patient_id) in enumerate(self.train_loader):
            data = data.to(self.config.DEVICE)
            target = target.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc
        }
        
        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)

    def train(self):
        """Main training loop with validation."""
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_metrics = self.evaluator.evaluate(self.val_loader)
            val_auc = val_metrics['auc']
            sensitivity = val_metrics['sensitivity']
            specificity = val_metrics['specificity']
            precision = val_metrics['precision']
            recall = val_metrics['recall']
            """         
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print("Early stopping triggered")
                break
            
            """
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val AUC = {val_auc:.4f} precision: {precision:.4f} recall {recall}")



class Evaluator():
    """Core evaluator with essential metrics."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate model with core metrics."""
        self.model.eval()
        predictions = []
        targets = []
        pdb.set_trace()
        for data, target, patient_id in data_loader:
            data = data.to(self.device)
            output = self.model(data)
            pred = torch.sigmoid(output)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate core metrics
        auc = roc_auc_score(targets, predictions)
        accuracy = ((predictions > 0.5) == targets).mean()
        
        # Calculate sensitivity and specificity
        true_positives = ((predictions > 0.5) & (targets == 1)).sum()
        false_negatives = ((predictions <= 0.5) & (targets == 1)).sum()
        true_negatives = ((predictions <= 0.5) & (targets == 0)).sum()
        false_positives = ((predictions > 0.5) & (targets == 0)).sum()
        
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)


        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'recall': recall,
            'precision': precision
        }


def main():
    config = TrainingConfig()


    # Code Requires ./dataprocessed mri_data repository
    pp = Preprocessor()
    X, y, ids = pp.transform()

    dataset = LGEDataset(X, y, ids)


    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        stratify = labels, 
        test_size = 0.3
    )


    test_labels = [dataset[i][1] for i in test_idx]


    val_idx, final_test_idx = train_test_split(
        test_idx,
        stratify = test_labels, 
        test_size = 0.5
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, final_test_idx)

    def count_classes(dataset):
        count1 = 0
        count0 = 0
        for element in range(len(dataset)):
            _, label, _ = dataset[element]

            if label == 1: 
                count1 += 1
            elif label == 0:
                count0 += 1
        print(f"{count1} + {count0}")

    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size = config.batch_size)


    # Get dataset + split
    model =  BaselineScorer()

    evaluator = Evaluator(model, config.DEVICE)

    trainer = Trainer(model, train_dataloader, evaluator, val_dataloader, config)
    

    trainer.train()

    return



if __name__ == '__main__':
   main() 