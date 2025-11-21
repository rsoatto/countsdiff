"""
Command-line interface for SNP diffusion package
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import numpy as np

# Handle both direct execution and module execution
try:
    from .training.trainer import CountsdiffTrainer
    from .config.config import Config
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from countsdiff.training.trainer import CountsdiffTrainer
    from countsdiff.config.config import Config


def train_command(args):
    """Execute training command"""
    # Load configuration
    config = Config.load_from_file(args.config)
    
    # Create and run trainer
    trainer = CountsdiffTrainer(config, device=args.device)
    trainer.train()

def continue_command(args):
    """Continue training from a checkpoint"""
    # Load configuration
    config = Config.load_from_neptune(args.run_id, project_name='countsdiff-iclr/ICLR')
    extra_steps = args.extra_steps if hasattr(args, 'extra_steps') else 0
    if extra_steps > 0:
        config['training']['n_steps'] += extra_steps
        print(f"Continuing training with {extra_steps} extra steps, total now {config['training']['n_steps']}")
    if args.override_config:
        override_config = Config.load_from_file(args.override_config)
        config.update(override_config)
        print(f"Overriding configuration with {args.override_config}")
    trainer = CountsdiffTrainer(config, device=args.device, run_id=args.run_id)
    trainer.train()
    


def train_cli():
    """CLI entry point for training"""
    parser = argparse.ArgumentParser(description="Train SNP/CIFAR-10 Blackout Diffusion Model")
    parser.add_argument('--config', required=True, help='Path to training configuration YAML')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        train_command(args)
    except KeyboardInterrupt:
        print("\nTraining cancelled by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)



def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SNP Blackout Diffusion - Hierarchical SNP Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train countsdiff
  countsdiff train --config configs/cifar10.yaml

  # Continue an existing training run
  countsdiff continue --run-id <run_id> --extra-steps 1000 --override-config configs/override.yaml
  
  # Train with specific device
  countsdiff train --config configs/snp_level0.yaml --device cuda:1
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a diffusion model')
    train_parser.add_argument('--config', required=True, help='Path to training configuration YAML')
    train_parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    train_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    train_parser.set_defaults(func=train_command)
    
    # Continue training command
    continue_parser = subparsers.add_parser('continue', help='Continue training from a checkpoint')
    continue_parser.add_argument('--run-id', required=True, help='Run ID of the previous training session') 
    continue_parser.add_argument('--extra-steps', type=int, default=0, help='Additional training steps to add')
    continue_parser.add_argument('--override-config', type=str, help='Path to YAML file to override configuration')
    continue_parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    continue_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    continue_parser.set_defaults(func=continue_command)
    
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
