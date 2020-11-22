import torch
from collections import namedtuple
from itertools import product

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_num_correct(preds, labels):
    """
    calculates the number of correct predictions.

    Args:
        preds: the predictions tensor with shape (batch_size, num_classes)
        labels: the labels tensor with shape (batch_size, num_classes)

    Returns:
        int: sum of correct predictions across the batch
    """
    return preds.argmax(dim=1).eq(labels).sum().item()


class RunBuilder():
    @staticmethod
    def get_runs(params):
        """
        build sets of parameters that define the runs.

        Args:
            params (OrderedDict): OrderedDict having hyper-parameter values

        Returns:
            list: containing list of all runs
        """
        Run = namedtuple('run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def get_all_preds(model, loader):
    """
    returns all the predictions of the entire dataset
    """
    all_preds = torch.tensor([])
    for batch in loader:
        images = batch[0].to(device)
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def get_mean_std(loader):
    """
    returns mean and std of a dataset
    """
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std
