#!/usr/bin/env python3
"""
Simple test script to debug LSTM reverse task
"""
import torch
import numpy as np
from simpletokenizers.simpletokenizers import CharTokenizer
from models.models import LSTM
from tasks.tasks import get_examples_for_task, compute_task_loss, compute_task_accuracy

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Initialize tokenizer
tok = CharTokenizer()
print(f"Vocab size: {len(tok.vocab)}")
print(f"Sample vocab: {list(tok.char_to_id.items())[:10]}")

# Create a simple test batch
batch_size = 4
seq_length = 5
task = "reverse"

# Generate a batch
batch_np = get_examples_for_task(task, tok, batch_size, seq_length, split='train')
print(f"\nBatch shape: {batch_np.shape}")

# Decode and print examples
for i in range(batch_size):
    decoded = tok.decode(batch_np[i])
    print(f"Example {i}: '{decoded}'")
    # Find the separator
    try:
        sep_idx = list(batch_np[i]).index(tok.char_to_id.get(" ", 0))
        input_part = tok.decode(batch_np[i][:sep_idx])
        output_part = tok.decode(batch_np[i][sep_idx+1:])
        print(f"  Input: '{input_part}' -> Expected output: '{output_part}'")
    except ValueError:
        print(f"  No separator found")

# Initialize LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

hidden_size = 128
num_heads = 4
head_size = hidden_size // num_heads

# Create embedding
embed = torch.nn.Embedding(len(tok.vocab), hidden_size, device=device, dtype=torch.bfloat16)

model = LSTM(
    input_size=hidden_size,
    output_size=len(tok.vocab),
    hidden_size=hidden_size,
    memory_size=0,
    head_size=head_size,
    num_heads=num_heads,
    embed=embed,
    device=device,
    dtype=torch.bfloat16
)

print(f"\nModel initialized with hidden_size={hidden_size}")

# Convert batch to tensor and embed
batch_tensor = torch.from_numpy(batch_np).to(device)
x_emb = embed(batch_tensor)
print(f"Embedded input shape: {x_emb.shape}")

# Forward pass
with torch.no_grad():
    logits, _, hidden = model(x_emb, require_gradients=False)
    print(f"Logits shape: {logits.shape}")

    # Get predictions
    preds = logits.argmax(dim=-1)
    print(f"Predictions shape: {preds.shape}")

    # Decode predictions
    print("\nPredictions:")
    for i in range(batch_size):
        pred_text = tok.decode(preds[i].cpu().numpy())
        actual_text = tok.decode(batch_np[i])
        print(f"Example {i}:")
        print(f"  Actual:  '{actual_text}'")
        print(f"  Predicted: '{pred_text}'")

# Compute loss
with torch.enable_grad():
    x_emb_grad = embed(batch_tensor)
    logits_grad, _, _ = model(x_emb_grad, require_gradients=True)
    loss = compute_task_loss(logits_grad, batch_np, tok, task, verbose=True)
    accuracy = compute_task_accuracy(logits_grad, batch_np, tok, task, verbose=True)

    print(f"\nInitial loss: {loss.item():.4f}")
    print(f"Initial accuracy: {accuracy:.4f}")

# Try training for a few steps
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n" + "="*60)
print("Training for 10 steps on the same batch (should overfit):")
print("="*60)

for step in range(10):
    optimizer.zero_grad()

    # Forward pass
    x_emb_train = embed(batch_tensor)
    logits_train, _, _ = model(x_emb_train, require_gradients=True)

    # Compute loss
    loss = compute_task_loss(logits_train, batch_np, tok, task)
    accuracy = compute_task_accuracy(logits_train, batch_np, tok, task)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Step {step+1}: Loss={loss.item():.4f}, Accuracy={accuracy:.4f}")

# Final predictions
print("\n" + "="*60)
print("Final predictions after training:")
print("="*60)

with torch.no_grad():
    x_emb_final = embed(batch_tensor)
    logits_final, _, _ = model(x_emb_final, require_gradients=False)
    preds_final = logits_final.argmax(dim=-1)

    for i in range(batch_size):
        actual_text = tok.decode(batch_np[i])
        pred_text = tok.decode(preds_final[i].cpu().numpy())

        # Parse input/output
        try:
            sep_idx = list(batch_np[i]).index(tok.char_to_id.get(" ", 0))
            input_part = tok.decode(batch_np[i][:sep_idx])
            expected_output = tok.decode(batch_np[i][sep_idx+1:])

            # Get predicted output
            pred_output = tok.decode(preds_final[i][sep_idx+1:].cpu().numpy())

            print(f"\nExample {i}:")
            print(f"  Input: '{input_part}'")
            print(f"  Expected: '{expected_output}'")
            print(f"  Predicted: '{pred_output}'")
            print(f"  Match: {expected_output.strip() == pred_output.strip()}")
        except ValueError:
            print(f"Example {i}: No separator found")
