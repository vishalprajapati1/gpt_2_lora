import argparse

from torch.optim import Adam
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from utils import *
from train_utils import *
from model import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        prompt = "Who am I?"

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        
        optimizer = Adam(model.parameters(), lr=args.lr)
        train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader, optimizer, device=args.device, epochs=args.epochs)


        def plot(train, val, unit = "Loss"):
            plt.plot(range(1, len(train) + 1), train, label=f"Train {unit}")
            plt.plot(range(1, len(val) + 1), val, label=f"Validation {unit}")
            plt.xlabel("Epoch")
            plt.ylabel(unit)
            plt.title(f"Train and Validation {unit}")
            plt.legend()
            plt.savefig(f'plots/LoRA_{unit}.png')
            plt.close()
            
        plot(train_losses, val_losses, unit = "Loss")
        plot(train_accuracies, val_accuracies, unit = "Accuracy")
        model.save_trainable_params(args.model_path)
    else:
        print("Invalid mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    parser.add_argument("--LoRA_enable", type=bool, default=True, help="Need to enable LoRA or keep it same")
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Using device:", args.device)

    main(args)
