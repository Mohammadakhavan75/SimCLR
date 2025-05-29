import argparse
import torch
from train import SimCLRTrainer, SimCLRModel, create_dataloader
from linear_eval import LinearEvaluator, create_eval_dataloaders

def main():
    parser = argparse.ArgumentParser(description='SimCLR Implementation - Paper Exact')
    parser.add_argument('--mode', choices=['pretrain', 'linear_eval'], required=True,
                       help='Mode: pretrain or linear_eval')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet50'],
                       help='Base model architecture')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for linear evaluation')
    
    # Paper-exact configurations (optional overrides)
    parser.add_argument('--batch_size', type=int, help='Batch size (default: paper values)')
    parser.add_argument('--lr', type=float, help='Learning rate (default: paper values)')
    parser.add_argument('--temperature', type=float, help='Temperature (default: paper values)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (default: paper values)')
    parser.add_argument('--one_idx_class', type=int, default=None, help='Select only one class for training (default: None, use full dataset). For replicating SimCLR paper results, this should be None.')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'pretrain':
        print("Starting SimCLR pre-training with paper-exact configuration...")
        
        # Create model
        model = SimCLRModel(base_model=args.model, out_dim=128)
        
        # Create trainer with paper-exact settings
        trainer = SimCLRTrainer(
            model=model,
            device=device,
            dataset=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            temperature=args.temperature,
            epochs=args.epochs,
            one_idx_class=args.one_idx_class
        )
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset_name=args.dataset,
            batch_size=trainer.batch_size,
            one_idx_class=args.one_idx_class  # Set to None for full dataset, or specify a class index for filtering
        )
        
        # Train
        trainer.train(dataloader)
        
    elif args.mode == 'linear_eval':
        print("Starting linear evaluation...")
        
        if not args.checkpoint:
            raise ValueError("Checkpoint path required for linear evaluation")
        
        # Load pre-trained model
        model = SimCLRModel(base_model=args.model, out_dim=128)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print training configuration from checkpoint
        if 'config' in checkpoint:
            print("Model was trained with:")
            for key, value in checkpoint['config'].items():
                print(f"  {key}: {value}")
        
        # Create evaluation dataloaders
        train_loader, test_loader, num_classes = create_eval_dataloaders(args.dataset)
        
        # Create evaluator
        evaluator = LinearEvaluator(model, num_classes, device)
        
        # Train and evaluate
        best_acc = evaluator.train_linear_classifier(train_loader, test_loader, epochs=100)
        final_acc = evaluator.evaluate(test_loader)
        
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Final test accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    main()