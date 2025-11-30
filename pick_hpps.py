#!/usr/bin/env python3
"""
Pick best hyperparameter configuration from sweep results.
Primary method: fastest converged configuration (minimum elapsed_time)
Fallback: configuration with minimum final loss if none converged
"""
import csv
import json
import os
import sys


def pick_best_config(input_filename, output_json_filename, optimizer_name):
    """Pick best configuration: fastest converged, or min loss if none converged."""

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

    print(f"\n{optimizer_name} Results:")
    print(f"  Total configurations tested: {len(results)}")

    # Primary method: find converged configurations
    converged_results = [r for r in results if r['converged']]

    if converged_results:
        # Pick fastest converging configuration
        best_result = min(converged_results, key=lambda x: x['elapsed_time'])
        print(f"  Converged configurations: {len(converged_results)}")
        print(f"  ✓ BEST: Fastest converged configuration")

        # Save to JSON
        with open(output_json_filename, 'w') as f:
            json.dump(best_result, f, indent=2)

        print(f"  → Saved to {output_json_filename}")

        # Print details
        if 'num_perturbations' in best_result:
            # SPSA format
            print(f"    LR={best_result['lr']:.6f}, Pert={best_result['num_perturbations']}, "
                  f"Batch={best_result['batch_size']}")
        else:
            # Adam format
            print(f"    LR={best_result['lr']:.6f}, Batch={best_result['batch_size']}")

        print(f"    Converged in {best_result['elapsed_time']:.2f}s ({best_result['num_steps']} steps)")
        print(f"    Final loss: {best_result['final_loss']:.4f}, Min loss: {best_result['min_loss']:.4f}")

    else:
        # Fallback: pick configuration with minimum final loss
        print(f"  Converged configurations: 0")
        print(f"  ⚠ No configurations converged")
        print(f"  → Fallback: Picking configuration with minimum final loss")

        best_result = min(results, key=lambda x: x['final_loss'])

        # Save to JSON
        with open(output_json_filename, 'w') as f:
            json.dump(best_result, f, indent=2)

        print(f"  → Saved to {output_json_filename}")

        # Print details
        if 'num_perturbations' in best_result:
            # SPSA format
            print(f"    LR={best_result['lr']:.6f}, Pert={best_result['num_perturbations']}, "
                  f"Batch={best_result['batch_size']}")
        else:
            # Adam format
            print(f"    LR={best_result['lr']:.6f}, Batch={best_result['batch_size']}")

        print(f"    Time: {best_result['elapsed_time']:.2f}s ({best_result['num_steps']} steps)")
        print(f"    Final loss: {best_result['final_loss']:.4f}, Min loss: {best_result['min_loss']:.4f}")

    return True


def main():
    print("=" * 80)
    print("PICKING BEST HYPERPARAMETER CONFIGURATIONS")
    print("=" * 80)

    # Pick best SPSA configuration
    pick_best_config('sweep_hpps_spsa_results.csv', 'hpps_spsa.json', 'SPSA')

    # Pick best Adam configuration
    pick_best_config('sweep_hpps_adam_results.csv', 'hpps_adam.json', 'ADAM')

    print("\n" + "=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
