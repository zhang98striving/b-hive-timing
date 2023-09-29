import numpy as np
import termplotlib as tpl
from sklearn.metrics import roc_curve, auc
from scipy.special import softmax


def terminal_roc(predictions, truth, title=None):
    if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) < 1e-3:
        pass
    else:
        predictions = softmax(predictions, axis=-1)

    if len(predictions.shape) == 1:
        bvsl = predictions
    else:
        b_pred = predictions[:, :3].sum(axis=-1)
        l_pred = predictions[:, -2:].sum(axis=-1)
        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
    if len(np.unique(truth)) > 2:
        b_jets = (truth == 0) | (truth == 1) | (truth == 2)
        c_veto = truth != 3
    else:
        b_jets = truth
        c_veto = np.ones(truth.shape, dtype=bool)
    # c_jets = truth == 3
    # l_jets = (truth == 4) | (truth == 5)
    # summed_jets = b_jets + c_jets + l_jets

    # b_veto = (truth != 0) & (truth != 1) & (truth != 2)
    # l_veto = truth != 4

    fig = tpl.figure()
    for label, veto in zip(["b vs l"], [c_veto]):
        fpr, tpr, _ = roc_curve(b_jets[veto], bvsl[veto])
        fig.plot(
            tpr,
            fpr,
            width=90,
            height=30,
            xlim=(0.3, 1),
            ylim=(0.0001, 1),
            label=label,
            xlabel="b-id",
            title=title,
            extra_gnuplot_arguments=["set ylabel miss-id", "set logscale y"],
        )
    fig.show()
