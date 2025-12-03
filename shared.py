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
        full_seq = [BOS] + seq.tolist() + [SEP] + seq[::-1].tolist() + [EOS]

        x_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))
        y_batch.append(torch.tensor(full_seq, dtype=torch.long, device=device))

    x_ids = nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=PAD)
    y_ids = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=PAD)

    return x_ids, y_ids

def compute_reverse_loss(logits, targets, pad_id):
    """Compute cross-entropy loss, ignoring padding tokens."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    loss = nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        reduction='mean'
    )
    return loss

def compute_accuracy(model, x_ids, y_ids, sep_id, pad_id, chunk_size=32):
    """
    Compute accuracy using iterative teacher forcing (same as loss computation).
    This ensures accuracy is measured on the full autoregressive generation.
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

        # Compute accuracy only on output part (after SEP token)
        # Note: preds[b, i] predicts y_ids[b, i+1] (next-token prediction)
        total_correct = 0
        total_tokens = 0

        for b in range(B):
            sep_positions = (y_ids[b] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) == 0:
                continue
            sep_pos = sep_positions[0].item()

            # Check predictions after SEP token
            # Iterate through output positions (after SEP, before PAD)
            for i in range(sep_pos + 1, Ly):
                if y_ids[b, i] == pad_id:
                    break

                # Prediction at position (i-1) predicts token at position i
                pred_idx = i - 1
                if pred_idx >= 0 and pred_idx < preds.shape[1]:
                    if preds[b, pred_idx] == y_ids[b, i]:
                        total_correct += 1
                    total_tokens += 1

        return total_correct / max(total_tokens, 1)
