import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F

from models.deepjet import DeepJet
from dataset.dataset import getDataset
from training.training import perform_training
from plotting.plotting import plot_roc_curve, plot_losses

def inference(model, testdata):
    model.eval()
    input = []
    output = []
    for data in testdata:
    # data shape: (batch, input_dim, 1)
        x, y = data[:,:-1,:], data[:,-1,0]
        with torch.no_grad():
            pred = model(x).cpu().numpy()
            if len(output) == 0:
                output = pred
                input = data.cpu().numpy()
            else:
                output = np.append(output, pred, axis=0)
                input = np.append(input, data.cpu().numpy(), axis=0)

    np.save("input", input)
    np.save("output", output)
    return input, output

if __name__=="__main__":
    # Import rich for pretty printing
    from rich.console import Console
    c = Console()

    # Creating a dictionary to store hyperparameters
    config_dict = {}   
    config_dict["model"] = {}

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        c.print("[black on yellow]Warning:","No CUDA device available. Running on cpu...")
    config_dict["device"] = device

    dataset = getDataset(config_dict)
    training_data, test_data = random_split(dataset.to(device), [0.8, 0.2])
    training_data = DataLoader(training_data, batch_size=10000)
    test_data     = DataLoader(test_data, batch_size=10000)

    # Model Defintion
    print("Model definition")
    model = DeepJet(config_dict["model"]["feature_edges"]).to(device)

    # Training
    print("Start training")
    train_metrics, test_metrics = perform_training(model, training_data, test_data, config_dict ,nepochs=2)

    print("Training finished. Saving data...")
    save_dict = {
        "model": model.state_dict(),
        "training_data": training_data,
        "test_data": test_data
    }
    # torch.save(model.state_dict(), "model.pt")
    # torch.save(training_data, 'training_dataloader.pth')
    # torch.save(test_data, 'test_dataloader.pth')
    torch.save(save_dict, "model")
    np.save("config_dict", config_dict)
    np.savez("train_metrics", loss=train_metrics[:,0], acc=train_metrics[:,1], allow_pickle=True)
    np.savez("test_metrics", loss=test_metrics[:,0], acc=test_metrics[:,1], allow_pickle=True)

    #testdata inference:
    input, output = inference(model, test_data)

    #plot_roc_curve(input, output)
    #plot_losses(train_metrics[:,0], test_metrics[:,0])

    print("Done")