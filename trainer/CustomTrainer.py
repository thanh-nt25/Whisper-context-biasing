from transformers import Seq2SeqTrainer
from transformers.trainer_utils import PredictionOutput
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class CustomTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that incorporates weighted cross-entropy loss for contextual biasing.
    This allows the model to put more emphasis on specific words provided in the bias_spans.
    """
    
    def __init__(
        self, 
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        bias_weight=2.0,  # Weight factor for biased tokens
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.bias_weight = bias_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the weighted cross-entropy loss.
        Tokens that appear in bias_spans will have a higher weight in the loss function.
        """
        if "bias_spans" not in inputs:
            # Fall back to standard loss if no bias_spans are provided
            return super().compute_loss(model, inputs, return_outputs)
        
        # Extract bias spans
        bias_spans = inputs.pop("bias_spans")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get the standard loss
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("label_ids")
            
        # Create a mask for biased tokens
        bias_mask = self._create_bias_mask(logits, labels, bias_spans)
        
        # Compute weighted loss
        loss = self._compute_weighted_loss(logits, labels, bias_mask)
        
        return (loss, outputs) if return_outputs else loss
    
    def _create_bias_mask(self, logits, labels, bias_spans):
        """
        Create a mask that identifies tokens in the labels that are part of bias spans.
        
        Args:
            logits: Output logits from the model [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            bias_spans: Tensor of bias word token ids [batch_size, max_spans, max_span_len]
            
        Returns:
            A binary mask of shape [batch_size, seq_len] where 1 indicates a biased token
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        # Initialize mask with zeros
        bias_mask = torch.zeros_like(labels, dtype=torch.float)
        
        # Skip special placeholder values in bias_spans (50256 is used as padding)
        valid_mask = bias_spans != 50256
        
        # For each sample in the batch
        for i in range(batch_size):
            # For each potential span position in the labels
            for pos in range(seq_len):
                if labels[i, pos] == -100:  # Skip ignored positions
                    continue
                
                # Check if this token is part of any bias span
                for span_idx in range(bias_spans.size(1)):
                    span = bias_spans[i, span_idx]
                    valid_indices = valid_mask[i, span_idx]
                    
                    if not valid_indices.any():
                        continue  # Skip if this is just padding
                    
                    valid_span = span[valid_indices]
                    span_len = valid_span.size(0)
                    
                    # Look ahead to check if this position starts a matching span
                    if pos + span_len <= seq_len:
                        # Extract the sequence from labels
                        label_slice = labels[i, pos:pos+span_len]
                        # Create a mask for ignored positions
                        ignored_mask = label_slice == -100
                        
                        # If there are ignored positions, we can't reliably check for a match
                        if ignored_mask.any():
                            continue
                        
                        # Check if this sequence matches the bias span
                        if torch.all(label_slice == valid_span):
                            # Mark all tokens in this span as biased
                            bias_mask[i, pos:pos+span_len] = 1.0
        
        return bias_mask
    
    def _compute_weighted_loss(self, logits, labels, bias_mask):
        """
        Compute weighted cross-entropy loss with higher weights for biased tokens.
        
        Args:
            logits: Output logits from the model [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len]
            bias_mask: Binary mask indicating biased tokens [batch_size, seq_len]
            
        Returns:
            Weighted cross-entropy loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Reshape for loss computation
        logits_view = logits.view(-1, vocab_size)
        labels_view = labels.view(-1)
        bias_mask_view = bias_mask.view(-1)
        
        # Create weights based on bias_mask
        # Default weight is 1.0, bias tokens get self.bias_weight
        weights = torch.ones_like(labels_view, dtype=torch.float, device=device)
        weights = torch.where(bias_mask_view == 1.0, self.bias_weight * torch.ones_like(weights), weights)
        
        # Only consider non-ignored positions
        active_mask = labels_view != -100
        active_logits = logits_view[active_mask]
        active_labels = labels_view[active_mask]
        active_weights = weights[active_mask]
        
        # Compute cross entropy loss for each position
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        position_losses = loss_fct(active_logits, active_labels)
        
        # Apply weights to the losses
        weighted_losses = position_losses * active_weights
        
        # Compute mean loss
        loss = weighted_losses.mean()
        
        return loss
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step on a batch of inputs, and possibly compute the loss.
        We need to override this to handle bias_spans correctly.
        """
        # Remove bias_spans from inputs before passing to the model
        if "bias_spans" in inputs:
            bias_spans = inputs.pop("bias_spans")
            # Save a copy to restore later
            has_bias_spans = True
        else:
            has_bias_spans = False
            
        # Call parent's prediction_step
        outputs = super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )
        
        # Restore bias_spans if they were present
        if has_bias_spans:
            inputs["bias_spans"] = bias_spans
            
        return outputs