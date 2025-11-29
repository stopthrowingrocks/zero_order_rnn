#!/usr/bin/env python3
"""
Sort hyperparameter sweep results by final loss.
Reads sweep_hpps_spsa_results.csv and sweep_hpps_adam_results.csv
and creates sorted versions.
"""
import csv
import os
import sys


def sort_csv_by_final_loss(input_filename, output_filename):
    """Sort CSV file by final_loss column and save to new file."""

    if not os.path.exists(input_filename):
        print(f"Warning: {input_filename} not found, skipping...")
        return False

    # Read all results
    results = []
    with open(input_filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ['lr', 'epsilon', 'final_loss', 'min_loss', 'initial_loss',
                          'improvement', 'final_acc', 'elapsed_time']:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, KeyError):
                        pass
                elif key in ['num_perturbations', 'batch_size', 'num_steps']:
                    try:
                        row[key] = int(row[key])
                    except (ValueError, KeyError):
                        pass
                elif key == 'converged':
                    row[key] = row[key].lower() == 'true'
            results.append(row)

    if not results:
        print(f"Warning: {input_filename} is empty, skipping...")
        return False

    # Sort by final_loss (ascending - lower is better)
    sorted_results = sorted(results, key=lambda x: x['final_loss'])

    # Write sorted results
    with open(output_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"✓ Sorted {len(sorted_results)} results from {input_filename}")
    print(f"  → Saved to {output_filename}")
    print(f"  Best final loss: {sorted_results[0]['final_loss']:.4f}")

    # Print top 5
    print(f"\n  Top 5 configurations by final loss:")
    for i, result in enumerate(sorted_results[:5], 1):
        if 'name' in result:
            # Adam format
            print(f"    {i}. {result['name']}: LR={result['lr']}, Batch={result['batch_size']}, "
                  f"Loss={result['final_loss']:.4f}, Converged={result['converged']}")
        else:
            # SPSA format
            print(f"    {i}. LR={result['lr']:.6f}, Pert={result['num_perturbations']}, "
                  f"Batch={result['batch_size']}, Loss={result['final_loss']:.4f}, "
                  f"Converged={result['converged']}")

    return True


def main():
    print("=" * 80)
    print("SORTING HYPERPARAMETER SWEEP RESULTS BY FINAL LOSS")
    print("=" * 80)
    print()

    # Sort SPSA results
    if sort_csv_by_final_loss('sweep_hpps_spsa_results.csv',
                              'sweep_hpps_spsa_results_sorted.csv'):
        print()

    # Sort Adam results
    if sort_csv_by_final_loss('sweep_hpps_adam_results.csv',
                              'sweep_hpps_adam_results_sorted.csv'):
        print()

    print("=" * 80)
    print("SORTING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
