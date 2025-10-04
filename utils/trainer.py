import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import TRAIN_CONFIG, PROJECT_ROOT, MODEL_CONFIG, EVAL_CONFIG
import os
import logging
from evaluator import Evaluator

logger = logging.getLogger(__name__)

class Trainer:
    """
    model training, validation, and testing.
    """
    def __init__(self, model, train_loader, valid_loader, test_loader, optimizer, loss_fn, device, trie, special_tokens, semantic_id_map):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.trie = trie
        self.special_tokens = special_tokens
        
        self.item_to_semantic = semantic_id_map
        self.semantic_id_to_item = {v: k for k, v in semantic_id_map.items()}
        
        self.evaluator = Evaluator()
        
        # Parameters for early stopping
        self.patience = TRAIN_CONFIG['patience']
        self.best_metric = -1
        self.epochs_no_improve = 0
        self.best_model_path = os.path.join(PROJECT_ROOT, "saved_models", "best_dsr_rec_model.pth")
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

    def _prepare_batch(self, batch):
        history = batch['history'].to(self.device)
        target_item_ids_tensor = batch['target']

        valid_indices = []
        for i, item_id in enumerate(target_item_ids_tensor.tolist()):
            if item_id in self.item_to_semantic:
                valid_indices.append(i)
            else:
                logger.warning(f"Target item_id {item_id} not found in semantic map. Skipping this sample in the batch.")

        if not valid_indices:
            return None, None, None 

        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
        
        # Filter the batch data based on valid indices
        history = history.index_select(0, valid_indices_tensor)
        target_item_ids = target_item_ids_tensor.index_select(0, valid_indices_tensor.cpu()).tolist()

        target_semantic_ids = [
            (self.special_tokens['BOS'],) + self.item_to_semantic[item_id]
            for item_id in target_item_ids
        ]
        
        labels = [
            self.item_to_semantic[item_id] + (self.special_tokens['EOS'],)
            for item_id in target_item_ids
        ]
        
        # Padding
        decoder_input = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in target_semantic_ids], 
            batch_first=True, 
            padding_value=self.special_tokens['PAD']
        ).to(self.device)

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq) for seq in labels], 
            batch_first=True, 
            padding_value=self.special_tokens['PAD']
        ).to(self.device)
        
        return history, decoder_input, labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            if batch is None: continue
            
            history, decoder_input, labels = self._prepare_batch(batch)

            if history is None:
                continue

            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(history, decoder_input)
            
            # Calculate loss
            # logits: (B, T, V), labels: (B, T)
            # view(-1, logits.shape[-1]) -> (B*T, V)
            # view(-1) -> (B*T)
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader, top_k=50):
        self.model.eval()
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                if batch is None: continue
                
                history = batch['history'].to(self.device)
                ground_truth_items = batch['target'].tolist()

                # prediction 
                for i in range(history.size(0)):
                    # [1, seq_len]
                    user_history = history[i].unsqueeze(0) 
                    
                    # Greedy + Trie constraint
                    predicted_semantic_id = self.model.predict(
                        user_history,
                        max_len=MODEL_CONFIG['semantic_id_length'] + 1, 
                        trie=self.trie,
                        bos_token_id=self.special_tokens['BOS'],
                        eos_token_id=self.special_tokens['EOS']
                    )
                    
                    predicted_item = self.semantic_id_to_item.get(tuple(predicted_semantic_id))

                    if predicted_item is not None:
                        all_predictions.append([predicted_item])
                    else:
                        all_predictions.append([]) # An invalid ID was generated

                    all_ground_truth.append(ground_truth_items[i])
        
        metrics = self.evaluator.calculate_metrics(all_predictions, all_ground_truth)
        return metrics

    def train(self, epochs):
        """The complete training loop."""
        for epoch in range(1, epochs + 1):
            avg_train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f}")
            
            if epoch % TRAIN_CONFIG['eval_every_n_epochs'] == 0:
                metrics = self.evaluate(self.valid_loader)
                logger.info(f"--- Validation Epoch {epoch} ---")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")
                
                # Early stopping logic
                # Use Recall@10  as the key metric
                current_metric = metrics.get(f'Recall@{max(EVAL_CONFIG["k_values"])}', -1)
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                    logger.info(f"New best model saved to {self.best_model_path} with Recall@{max(EVAL_CONFIG['k_values'])}: {current_metric:.4f}")
                else:
                    self.epochs_no_improve += 1
                    logger.info(f"No improvement for {self.epochs_no_improve} validation steps. Patience: {self.patience}")

                if self.epochs_no_improve >= self.patience:
                    logger.info("Early stopping triggered.")
                    break
        
        logger.info("Training finished.")
        # Load the best model for final testing
        logger.info(f"Loading best model from {self.best_model_path} for final testing.")
        self.model.load_state_dict(torch.load(self.best_model_path))
        test_metrics = self.evaluate(self.test_loader)
        logger.info("--- Final Test Results ---")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")