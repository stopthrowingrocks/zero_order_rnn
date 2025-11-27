def main():
    base_seed = 42
    convergence_loss = 0.1

    # LR values from high to low (reversed from original)
    learning_rates = 10 ** np.arange(-0.6, -2 - 0.6, -0.2)  # 10^0 down to 10^-2

    # Start with generous max_time, will be updated after first convergence
    # Hard cap at 60 seconds per run
    initial_max_time = 60.0
    max_time = initial_max_time
    HARD_CAP_TIME = 60.0

    results = []
    first_convergence_idx = None
    first_convergence_time = None

    for lr_idx, lr in enumerate(learning_rates):
        # If we've found convergence and are 2+ LRs past it, stop
        if first_convergence_idx is not None and lr_idx >= first_convergence_idx + 2:
            print(f"  Stopping LR search (2 LRs past first convergence)")
            break

        seed = base_seed + lr_idx

        # Create wandb run name
        run_name = (f"fast_batch_{batch_size}_pert_{perturbations}_gpus_{num_gpus}_"
                   f"lr_{lr:.6f}_vocab_{vocab_size}_min_{min_tokens}_max_{int(max_tokens)}")

        print(f"    LR={lr:.6f} (max_time={max_time:.1f}s)...", end=" ", flush=True)

        # Initialize wandb
        wandb.init(
            project="zero_order_rnn",
            name=run_name,
            config={
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': num_gpus,
                'learning_rate': lr,
                'vocab_size': vocab_size,
                'min_tokens': min_tokens,
                'max_tokens': int(max_tokens),
                'seed': seed,
                'epsilon': lr,  # epsilon = learning_rate
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'max_time': max_time,
                'optalg': 'SPSA_adaptive'
            },
            reinit=True
        )

        try:
            # Returns None if intended to continue, otherwise returns final_state
            def end_condition(ctx):
                if ctx['loss'] < convergence_loss:
                    return 'converged'
                if ctx['time_elapsed'] < max_time:
                    return 'time_elapsed'
                if ctx['loss'] > 3.0 * obj['losses'][0]:
                    return 'max_loss'
                return None
            def model_initializer(device):
                embed = nn.Embedding(vocab_size, hidden_size, device=device, dtype=torch.bfloat16)
                model = LSTM(
                    input_size=hidden_size,
                    output_size=vocab_size,
                    hidden_size=hidden_size,
                    memory_size=0,
                    head_size=hidden_size // num_heads,
                    num_heads=num_heads,
                    embed=embed,
                    device=device,
                    dtype=torch.bfloat16
                )
                return model
            def batch_generator():
                return generate_reverse_batch(
                    batch_size, min_tokens, max_tokens, vocab_size, device_obj
                )
            result = train_to_condition(
                batch_size=batch_size,
                perturbations=perturbations,
                num_gpus=num_gpus,
                learning_rate=lr,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                max_tokens=int(max_tokens),
                device=device,
                seed=seed,
                end_condition=end_condition,
            )

            # Log final results to wandb
            wandb.log({
                'final_loss': result['final_loss'],
                'total_steps': result['steps'],
                'total_time': result['elapsed_time'],
                'terminated_early': result['terminated_early'],
                'converged': result['converged']
            })

            # Store result
            result['lr'] = lr
            result['max_tokens'] = int(max_tokens)
            result['batch_size'] = batch_size
            result['perturbations'] = perturbations
            result['max_time_used'] = max_time
            results.append(result)

            # Check if this is first convergence
            if result['converged'] and first_convergence_idx is None:
                first_convergence_idx = lr_idx
                first_convergence_time = result['elapsed_time']
                # Double the time for remaining searches, but hard cap at 30s
                max_time = min(first_convergence_time * 2.0, HARD_CAP_TIME)
                print(f"CONVERGED in {result['elapsed_time']:.2f}s! Setting max_time={max_time:.1f}s")
            elif result['converged']:
                print(f"Converged in {result['elapsed_time']:.2f}s")
            elif result['terminated_early']:
                print(f"Terminated early (3x loss increase)")
            else:
                print(f"Did not converge (loss={result['final_loss']:.4f})")

            # Prepare and append CSV row
            csv_row = {
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': num_gpus,
                'learning_rate': lr,
                'vocab_size': vocab_size,
                'min_tokens': min_tokens,
                'max_tokens': int(max_tokens),
                'seed': seed,
                'final_loss': result['final_loss'],
                'steps': result['steps'],
                'elapsed_time': result['elapsed_time'],
                'converged': result['converged'],
                'terminated_early': result['terminated_early'],
                'max_time_used': max_time,
                'optalg': 'SPSA_adaptive'
            }
            append_to_csv('losses_fast.csv', csv_row)

        except Exception as e:
            print(f"ERROR: {e}")

            # Log error to CSV with NaN values
            csv_row = {
                'batch_size': batch_size,
                'perturbations': perturbations,
                'num_gpus': num_gpus,
                'learning_rate': lr,
                'vocab_size': vocab_size,
                'min_tokens': min_tokens,
                'max_tokens': int(max_tokens),
                'seed': seed,
                'final_loss': float('nan'),
                'steps': 0,
                'elapsed_time': 0.0,
                'converged': False,
                'terminated_early': False,
                'max_time_used': max_time,
                'optalg': 'SPSA_adaptive'
            }
            append_to_csv('losses_fast.csv', csv_row)

        finally:
            wandb.finish()

    return results
