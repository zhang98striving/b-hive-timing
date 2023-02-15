from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curve(input, output):
    b_jets = (input[:,-1,0] == 0) | (input[:,-1,0] == 1) | (input[:,-1,0] == 2)
    print(b_jets[:10])

    prob_b = output[:,:3].sum(axis=1)
    print(prob_b.shape, prob_b[:10])

    c_veto = (input[:,-1,0]!=3) & (input[:,0,0] > 30) # id == 3 + jet_pt > 30
    light_veto = ((input[:,-1,0]!=4) & (input[:,-1,0]!=5)) & (input[:,0,0] > 30) # id!=4 or !=5 + jet_pt>30
    print(c_veto.sum(), light_veto.sum())
    print((input[:,0,0] > 30).sum(), light_veto.sum())

    fpr, tpr, _ = roc_curve(b_jets[c_veto], prob_b[c_veto])
    plt.plot(tpr, fpr, label="c")

    fpr, tpr, _ = roc_curve(b_jets[light_veto], prob_b[light_veto])
    plt.plot(tpr, fpr, label="udsg")

    plt.legend()
    plt.title("pt>30GeV, tt events")
    plt.xlabel("b jet efficiency")
    plt.ylabel("misid. probability")
    plt.savefig("roc.pdf")
    plt.close()

def plot_losses(train_loss, test_loss):
    plt.title("Losses")
    plt.plot(*np.array(list(enumerate(test_loss))).T, label="Test")
    plt.plot(*np.array(list(enumerate(train_loss))).T, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.pdf")
    plt.close()