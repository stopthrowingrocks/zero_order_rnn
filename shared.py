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

def generate_predictions(model, x_ids, y_ids, require_gradients, chunk_size=32):
    """
    Generate predictions for y_ids using teacher forcing.

    Args:
        model: The LSTM model
        x_ids: Input token IDs [batch_size, seq_len]
        y_ids: Target token IDs [batch_size, seq_len]
        require_gradients: Whether to compute gradients
        chunk_size: Size of chunks for processing long sequences

    Returns:
        logits: Predictions for all y_ids tokens [batch_size, Ly, vocab_size]
    """
    # Conditionally disable gradients
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

        # Process input sequence first and collect all logits
        all_logits = []
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory, require_gradients=require_gradients)
            hidden = hidden_new
            memory = mem_new
            # Save the last logit from x processing - it predicts y_ids[:, 0]
            if chunk_end == Lx:
                # This is the last chunk, take the final logit (after SEP token)
                first_y_pred = out_chunk[:, -1:, :]  # [B, 1, vocab_size]
                all_logits.append(first_y_pred)
            pos = chunk_end

        # Now process target sequence chunk by chunk and collect remaining predictions
        # Feed y_ids[:, 0:Ly-1] to get predictions for y_ids[:, 1:Ly]
        pos = 0
        while pos < Ly - 1:
            chunk_end = min(pos + chunk_size, Ly - 1)
            y_chunk = y_ids[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)

            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory, require_gradients=require_gradients)
            hidden = hidden_new
            memory = mem_new

            # Collect logits for this chunk
            all_logits.append(out_chunk)

            pos = chunk_end

        # Concatenate all logits
        if all_logits:
            logits = torch.cat(all_logits, dim=1)  # [B, Ly, vocab_size]
        else:
            # Return empty logits if no predictions were made
            B = x_ids.shape[0]
            vocab_size = model.output_size
            logits = torch.zeros(B, 0, vocab_size, device=x_ids.device, dtype=next_param.dtype)

        return logits

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
    # Generate predictions for all y_ids tokens
    logits = generate_predictions(model, x_ids, y_ids, require_gradients, chunk_size)

    # Compute loss
    # Always use float32 for loss accumulation to avoid bfloat16 precision issues
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=x_ids.device)
    total_predicted_tokens = 0

    Ly = y_ids.shape[1]
    B = x_ids.shape[0]

    # Compute loss for all predictions
    # logits[:, i] predicts y_ids[:, i]
    for i in range(Ly):
        if i < logits.shape[1]:
            pred = logits[:, i, :]  # [B, vocab_size]
            target = y_ids[:, i]  # [B]

            # Compute loss for this position
            loss = criterion(pred, target)
            total_loss += loss.float() * B
            total_predicted_tokens += B

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
    # Generate predictions for all y_ids tokens
    logits = generate_predictions(model, x_ids, y_ids, require_gradients=False, chunk_size=chunk_size)

    if logits.shape[1] == 0:
        return 0.0

    # Get predictions
    preds = logits.argmax(dim=-1)  # [B, Ly]

    B = x_ids.shape[0]
    Ly = y_ids.shape[1]

    # Compute accuracy on entire y_ids sequence (excluding PAD)
    # Note: preds[b, i] predicts y_ids[b, i] directly
    total_correct = 0
    total_tokens = 0

    for b in range(B):
        # Iterate through all target positions (before PAD)
        for i in range(Ly):
            if y_ids[b, i] == pad_id:
                break

            # Prediction at position i predicts token at position i
            if i < preds.shape[1]:
                if preds[b, i] == y_ids[b, i]:
                    total_correct += 1
                total_tokens += 1

    return total_correct / max(total_tokens, 1)
