import numpy

from chainer.datasets.sub_dataset import SubDataset


class PaddableSubDataset(SubDataset):

    def pad_labels(self, new_label_length, pad_value):
        self._dataset.pad_labels(new_label_length, pad_value)


def split_dataset(dataset, split_at, order=None):
    """Splits a dataset into two subsets.
    This function creates two instances of :class:`SubDataset`. These instances
    do not share any examples, and they together cover all examples of the
    original dataset.
    Args:
        dataset: Dataset to split.
        split_at (int): Position at which the base dataset is split.
        order (sequence of ints): Permutation of indexes in the base dataset.
            See the document of :class:`SubDataset` for details.
    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset represents the
            examples of indexes ``order[:split_at]`` while the second subset
            represents the examples of indexes ``order[split_at:]``.
    """
    n_examples = len(dataset)
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at >= n_examples:
        raise ValueError('split_at exceeds the dataset size')
    subset1 = PaddableSubDataset(dataset, 0, split_at, order)
    subset2 = PaddableSubDataset(dataset, split_at, n_examples, order)
    return subset1, subset2


def split_dataset_random(dataset, first_size, seed=None):
    """Splits a dataset into two subsets randomly.
    This function creates two instances of :class:`SubDataset`. These instances
    do not share any examples, and they together cover all examples of the
    original dataset. The split is automatically done randomly.
    Args:
        dataset: Dataset to split.
        first_size (int): Size of the first subset.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset contains
            ``first_size`` examples randomly chosen from the dataset without
            replacement, and the second subset contains the rest of the
            dataset.
    """
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return split_dataset(dataset, first_size, order)

def split_dataset_n(dataset, n, order=None):
    """Splits a dataset into ``n`` subsets.
    Args:
        dataset: Dataset to split.
        n(int): The number of subsets.
        order (sequence of ints): Permutation of indexes in the base dataset.
            See the document of :class:`SubDataset` for details.
    Returns:
        list: List of ``n`` :class:`SubDataset` objects.
            Each subset contains the examples of indexes
            ``order[i * (len(dataset) // n):(i + 1) * (len(dataset) // n)]``
            .
    """
    n_examples = len(dataset)
    sub_size = n_examples // n
    return [PaddableSubDataset(dataset, sub_size * i, sub_size * (i + 1), order)
            for i in range(n)]


def split_dataset_n_random(dataset, n, seed=None):
    """Splits a dataset into ``n`` subsets randomly.
    Args:
        dataset: Dataset to split.
        n(int): The number of subsets.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
    Returns:
        list: List of ``n`` :class:`SubDataset` objects.
            Each subset contains ``len(dataset) // n`` examples randomly chosen
            from the dataset without replacement.
    """
    n_examples = len(dataset)
    sub_size = n_examples // n
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return [PaddableSubDataset(dataset, sub_size * i, sub_size * (i + 1), order)
            for i in range(n)]
