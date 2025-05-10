import torch
import time

import matplotlib.pyplot as plt

def plot(train, val, n, unit = "Loss"):
    plt.plot(range(1, len(train) + 1), train, label=f"Train {unit}")
    plt.plot(range(1, len(val) + 1), val, label=f"Validation {unit}")
    plt.xlabel("Epoch")
    plt.ylabel(unit)
    plt.title(f"Train and Validation {unit}")
    plt.legend()
    plt.savefig(f'plots/LoRA_{unit}_at_{n}.png')
    plt.close()

def train(model, train_loader, val_loader, optimizer, device, epochs=10):

    print_time = True
    print("\n\n Started training the model.\n")

    train_losses    = []
    val_losses      = []
    train_accs      = []
    val_accs        = []

    plot_at_checkpoint = False
    plot_checkpoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    for epoch in range(epochs):

        start = time.time()
        model.train()
        train_loss , train_correct, train_total = 0, 0, 0

        for input_batch, mask_batch, label_batch in train_loader:

            input_batch = input_batch.to(device)
            mask_batch = mask_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = model(input_batch, mask_batch)
            loss = torch.nn.functional.cross_entropy(outputs, label_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += label_batch.size(0)
            train_correct += (predicted == label_batch).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * (train_correct / train_total)

        # computing validation loss and accuracy
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for input_batch, mask_batch, label_batch in val_loader:
                input_batch = input_batch.to(device)
                mask_batch = mask_batch.to(device)
                label_batch = label_batch.to(device)


                outputs = model(input_batch, mask_batch)
                loss = torch.nn.functional.cross_entropy(outputs, label_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += label_batch.size(0)
                val_correct += (predicted == label_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\n\nCompleted Epoch {epoch + 1} of {epochs}:")
        end = time.time()
        if print_time: print(f"It took : {end - start:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if epoch + 1 in plot_checkpoints and plot_at_checkpoint:
            plot(train_losses, val_losses, epoch + 1, "Loss")
            plot(train_accs, val_accs, epoch + 1, "Accuracy")


    return train_losses, val_losses, train_accs, val_accs
