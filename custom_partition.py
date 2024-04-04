import logging
import numpy as np

logger = logging.getLogger()

def assign_examples_to_clients(y_train, n_parties, partition):
    """

    :param y_train: labels in the training set (distributed among clients)
    :param n_parties: number of clients
    :param partition: type of partition
    :return: dict with client indexes as keys and lists of indexes of assigned examples as values
    """

    # Homogeneous (although not class-balanced) distribution by default -- copies and pasted from utils,partition_data
    logger.info(f"Running custom partitioning: {partition}")
    n_train = len(y_train)
    idxs = np.random.permutation(n_train)
    batch_idxs = np.array_split(idxs, n_parties)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
    return net_dataidx_map
