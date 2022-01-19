import torch
from torch.utils.data.dataset import Dataset

class MyTabularDataset(Dataset):

    def __init__(self, dataset, label_col):
        """
        :param dataset: 数据
        :param label_col: 标签列名
        """

        self.label = torch.LongTensor(dataset[label_col].values)

        self.data = dataset.drop(columns=[label_col]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.label[index]
        data = self.data[index]

        return torch.Tensor(data), label

