import torch
import torch.nn as nn
import numpy as np

def train_model(dataloader, model, loss_fn, optimizer):
    losses = []
    accuracy = 0.0
    model.train()
    for data in dataloader:
        x, y = data[:,:-1,:], data[:,-1,0]
        pred = model(x)
        loss = loss_fn(pred, y.type(torch.LongTensor).to("cuda"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)

def test_model(dataloader, model, loss_fn):
    losses = []
    accuracy = 0.0
    model.eval()
    for data in dataloader:
        x, y = data[:,:-1,:], data[:,-1,0]
        with torch.no_grad():
            pred = model(x)
            loss = loss_fn(pred, y.type(torch.LongTensor).to("cuda"))
            losses.append(loss.cpu().numpy())
            accuracy += torch.sum(y == pred.argmax(dim=1))
    accuracy /= len(dataloader.dataset)
    print("  ", np.array(losses).mean(), float(accuracy))
    return np.array(losses).mean(), float(accuracy)

def perform_training(model, training_data, test_data, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    nepochs = kwargs["nepochs"]
    train_metrics = np.zeros((nepochs, 2))
    test_metrics  = np.zeros((nepochs, 2))
    for t in range(nepochs):
        print(t, "of", nepochs)
        loss, acc = train_model(training_data, model, loss_fn, optimizer)
        train_metrics[t,:] = np.array([loss, acc])
        loss, acc = test_model(test_data, model, loss_fn)
        test_metrics[t,:]  = np.array([loss, acc])

    return train_metrics, test_metrics