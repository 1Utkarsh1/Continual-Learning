#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Continual Learning System.
This script orchestrates the training and evaluation of continual learning approaches.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
import random
from datetime import datetime

# Add the 'src' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import get_task_sequence
from src.models.model_factory import get_model
from src.methods.baseline import BaselineLearner
from src.methods.ewc import EWCLearner
from src.methods.replay import ExperienceReplayLearner
from src.methods.lwf import LwFLearner
from src.utils.metrics import evaluate_performance, compute_forgetting
from src.utils.visualization import plot_performance, plot_forgetting


def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'continual_learning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Continual Learning System')
    
    parser.add_argument('--method', type=str, required=True,
                        choices=['baseline', 'ewc', 'replay', 'lwf'],
                        help='Continual learning method to use')
    
    parser.add_argument('--tasks', type=str, default='mnist_split',
                        help='Predefined task sequence (or path to custom YAML file)')
    
    parser.add_argument('--model', type=str, default='simple_cnn',
                        help='Base model architecture')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs per task')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    # EWC specific arguments
    parser.add_argument('--lambda_ewc', type=float, default=5000,
                        help='Regularization strength for EWC')
    parser.add_argument('--fisher_sample_size', type=int, default=200,
                        help='Number of samples to estimate Fisher information')
    
    # Experience Replay specific arguments
    parser.add_argument('--buffer_size', type=int, default=500,
                        help='Size of the replay buffer')
    parser.add_argument('--replay_batch_size', type=int, default=32,
                        help='Batch size for replayed samples')
    
    # LwF specific arguments
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for knowledge distillation in LwF')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for distillation loss in LwF')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Frequency of evaluation during training (in epochs)')
    
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()


def load_task_config(task_name_or_path):
    """
    Load task configuration from predefined sequences or a custom YAML file.
    
    Args:
        task_name_or_path (str): Name of predefined task or path to custom YAML file
        
    Returns:
        dict: Task configuration
    """
    predefined_tasks = {
        'mnist_split': [
            {'name': 'mnist_0_4', 'dataset': 'mnist', 'classes': [0, 1, 2, 3, 4]},
            {'name': 'mnist_5_9', 'dataset': 'mnist', 'classes': [5, 6, 7, 8, 9]}
        ],
        'multi_dataset': [
            {'name': 'mnist', 'dataset': 'mnist', 'classes': 'all'},
            {'name': 'fashion_mnist', 'dataset': 'fashion_mnist', 'classes': 'all'},
            {'name': 'kmnist', 'dataset': 'kmnist', 'classes': 'all'}
        ],
    }
    
    if task_name_or_path in predefined_tasks:
        return {'task_sequence': predefined_tasks[task_name_or_path]}
    
    # Load from YAML file
    if os.path.exists(task_name_or_path):
        with open(task_name_or_path, 'r') as f:
            return yaml.safe_load(f)
    
    raise ValueError(f"Task sequence '{task_name_or_path}' not recognized and file not found.")


def run_continual_learning(args, logger):
    """
    Run the continual learning experiment.
    
    Args:
        args: Command line arguments
        logger: Logger instance
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load task sequence
    task_config = load_task_config(args.tasks)
    task_sequence = task_config['task_sequence']
    
    logger.info(f"Task sequence: {[task['name'] for task in task_sequence]}")
    
    # Create data loaders for all tasks
    task_data = get_task_sequence(task_sequence, args.batch_size)
    
    # Initialize model and learner
    input_shape = task_data[0]['train_loader'].dataset[0][0].shape
    all_classes = set()
    for task in task_sequence:
        if task['classes'] == 'all':
            # 'all' is a sentinel meaning all classes in the dataset (e.g. 10 for MNIST variants)
            all_classes.update(range(10))
        else:
            all_classes.update(task['classes'])
    num_classes = len(all_classes)
    
    model = get_model(args.model, input_shape, num_classes)
    model = model.to(device)
    
    # Select appropriate learner based on method
    if args.method == 'baseline':
        learner = BaselineLearner(
            model=model,
            device=device,
            learning_rate=args.learning_rate
        )
    elif args.method == 'ewc':
        learner = EWCLearner(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            lambda_ewc=args.lambda_ewc,
            fisher_sample_size=args.fisher_sample_size
        )
    elif args.method == 'replay':
        learner = ExperienceReplayLearner(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            replay_batch_size=args.replay_batch_size
        )
    elif args.method == 'lwf':
        learner = LwFLearner(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            alpha=args.alpha
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Create directory to save results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.save_dir)
    experiment_dir = os.path.join(results_dir, f"{args.method}_{args.tasks}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Track performance after each task
    performance_matrix = np.zeros((len(task_sequence), len(task_sequence)))
    
    # Sequentially train on each task
    for task_id, task_data_dict in enumerate(task_data):
        task_name = task_data_dict['name']
        train_loader = task_data_dict['train_loader']
        val_loader = task_data_dict['val_loader']
        
        logger.info(f"Starting training on task {task_id+1}/{len(task_data)}: {task_name}")
        
        # Train on current task
        learner.train(
            train_loader=train_loader,
            val_loader=val_loader,
            task_id=task_id,
            epochs=args.epochs,
            eval_freq=args.eval_freq
        )
        
        # Evaluate on all tasks seen so far
        for eval_task_id in range(task_id + 1):
            eval_task_data = task_data[eval_task_id]
            eval_loader = eval_task_data['test_loader']
            
            accuracy = learner.evaluate(eval_loader, eval_task_id)
            performance_matrix[task_id, eval_task_id] = accuracy
            
            logger.info(f"After task {task_id+1}, accuracy on task {eval_task_id+1}: {accuracy:.2f}%")
    
    # Calculate forgetting
    forgetting_matrix = compute_forgetting(performance_matrix)
    
    # Save performance metrics
    np.save(os.path.join(experiment_dir, 'performance_matrix.npy'), performance_matrix)
    np.save(os.path.join(experiment_dir, 'forgetting_matrix.npy'), forgetting_matrix)
    
    # Save model
    learner.save(os.path.join(experiment_dir, 'final_model.pt'))
    
    # Generate plots
    task_names = [task['name'] for task in task_sequence]
    performance_plot = plot_performance(performance_matrix, task_names)
    forgetting_plot = plot_forgetting(forgetting_matrix, task_names)
    
    performance_plot.savefig(os.path.join(experiment_dir, 'performance.png'))
    forgetting_plot.savefig(os.path.join(experiment_dir, 'forgetting.png'))
    
    # Print summary
    logger.info("\nExperiment summary:")
    logger.info(f"Method: {args.method}")
    logger.info(f"Task sequence: {[task['name'] for task in task_sequence]}")
    logger.info(f"Average final accuracy: {np.mean(performance_matrix[-1, :]):.2f}%")
    logger.info(f"Average forgetting: {np.mean(forgetting_matrix[-1, :]):.2f}%")
    logger.info(f"Results saved to: {experiment_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Continual Learning experiment")
    logger.info(f"Arguments: {args}")
    
    # Run experiment
    try:
        run_continual_learning(args, logger)
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.exception(f"Experiment failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 