import argparse
import torch
from src.model import ContextualReasoningModel

def train(args):
    model = ContextualReasoningModel(model_name=args.model_name, num_tasks=args.num_tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Multi-task contextual reasoning training loop (see paper for details)
    print(f"Training {args.model_name} on {args.num_tasks} tasks with lr={args.lr}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
