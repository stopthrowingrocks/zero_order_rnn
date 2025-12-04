import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from models.models import LSTM

def generate_reverse_batch(batch_size, min_seq_length, max_seq_length, vocab_size, device='cuda'):
    """Generate a batch of reverse sequences with variable lengths."""
    BOS = vocab_size - 4
    SEP = vocab_size - 3
    EOS = vocab_size - 2
    PAD = vocab_size - 1
    content_vocab_size = vocab_size - 4

    x_batch = []
    y_batch = []

    for _ in range(batch_size):
        seq_length = np.random.randint(min_seq_length, max_seq_length + 1)
        seq = np.random.randint(0, content_vocab_size, size=seq_length)
        x_seq = [BOS] + seq.tolist() + [SEP]
        y_seq = seq[::-1].tolist() + [EOS]

        x_batch.append(torch.tensor(x_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(y_seq, dtype=torch.long, device=device))

    x_ids = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_ids = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=PAD)

    return x_ids, y_ids

def compute_loss(model, x_ids, y_ids, criterion, require_gradients, chunk_size=32):
    """
    Unified teacher forcing loss computation for both Adam and SPSA.

    Args:
        model: The LSTM model
        x_ids: Input token IDs [batch_size, seq_len]
        y_ids: Target token IDs [batch_size, seq_len]
        criterion: Loss criterion (e.g., CrossEntropyLoss)
        require_gradients: Whether to compute gradients (True for Adam, False for SPSA)
        chunk_size: Size of chunks for processing long sequences

    Returns:
        avg_loss: Average loss as a torch.Tensor (float32)
    """
    # Conditionally disable gradients for SPSA
    context = torch.no_grad() if not require_gradients else torch.enable_grad()

    with context:
        x_emb = model.embed(x_ids)

        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)
        Lx = x_emb.shape[1]
        Ly = y_ids.shape[1]

        hidden = None
        memory = None
        # Always use float32 for loss accumulation to avoid bfloat16 precision issues
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=x_ids.device)
        total_predicted_tokens = 0

        # Process input sequence first
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=require_gradients)
            hidden = hidden_new
            memory = mem_new
            pos = chunk_end

        # Now process target sequence chunk by chunk
        pos = 0
        while pos < Ly - 1:  # -1 because we don't embed the last target token
            chunk_end = min(pos + chunk_size, Ly - 1)
            # Only embed the current chunk of target sequence
            y_chunk = y_ids[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=require_gradients)

            # Update states
            hidden = hidden_new
            memory = mem_new

            # Compute loss for this chunk
            out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
            targets = y_ids[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction

            if targets.size(0) > 0:  # ensure we have targets
                chunk_loss = criterion(out_chunk, targets)
                # Always convert to float32 before accumulation to avoid bfloat16 precision issues
                total_loss += chunk_loss.float() * targets.size(0)
                total_predicted_tokens += targets.size(0)

            pos = chunk_end

        if total_predicted_tokens == 0:
            # Return float32 tensor with gradients if needed
            return torch.tensor(0.0, dtype=torch.float32, device=x_ids.device, requires_grad=require_gradients)

        avg_loss = total_loss / total_predicted_tokens
        # Ensure return is always float32 tensor
        if avg_loss.dtype != torch.float32:
            avg_loss = avg_loss.float()
        return avg_loss

def compute_accuracy(model, x_ids, y_ids, pad_id, chunk_size=32):
    """
    Compute accuracy using iterative teacher forcing (same as loss computation).
    This ensures accuracy is measured on the full autoregressive generation.

    Note: With the new data format, y_ids contains only the target output (no SEP).
    The entire y_ids sequence (excluding PAD) is evaluated for accuracy.
    """
    with torch.no_grad():
        x_emb = model.embed(x_ids)

        next_param = next(model.parameters())
        if x_emb.dtype != next_param.dtype:
            x_emb = x_emb.to(dtype=next_param.dtype)
        Lx = x_emb.shape[1]
        Ly = y_ids.shape[1]

        B = x_ids.shape[0]

        hidden = None
        memory = None

        # Process input sequence first
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=False)
            hidden = hidden_new
            memory = mem_new
            pos = chunk_end

        # Now process target sequence chunk by chunk and collect predictions
        all_logits = []
        pos = 0
        while pos < Ly - 1:
            chunk_end = min(pos + chunk_size, Ly - 1)
            y_chunk = y_ids[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=False)
            hidden = hidden_new
            memory = mem_new

            # Collect logits for this chunk
            all_logits.append(out_chunk)

            pos = chunk_end

        # Concatenate all logits
        if all_logits:
            logits = torch.cat(all_logits, dim=1)  # [B, Ly-1, vocab_size]
        else:
            return 0.0

        # Get predictions
        preds = logits.argmax(dim=-1)  # [B, Ly-1]

        # Compute accuracy on entire y_ids sequence (excluding PAD)
        # Note: preds[b, i] predicts y_ids[b, i+1] (next-token prediction)
        total_correct = 0
        total_tokens = 0

        for b in range(B):
            # Iterate through all target positions (before PAD)
            for i in range(Ly):
                if y_ids[b, i] == pad_id:
                    break

                # Prediction at position (i-1) predicts token at position i
                pred_idx = i - 1
                if pred_idx >= 0 and pred_idx < preds.shape[1]:
                    if preds[b, pred_idx] == y_ids[b, i]:
                        total_correct += 1
                    total_tokens += 1

        return total_correct / max(total_tokens, 1)
