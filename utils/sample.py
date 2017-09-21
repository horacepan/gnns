import torch
from torch.utils.data import Dataset
import pdb

class Sample(Dataset):
    def __init__(self, size):
        self.tensors = [torch.eye(i) for i in range(1, size+1)]
        self.targets = [i for i in range(size)]
    def __getitem__(self, index):
        return self.tensors[index], self.targets[index]
    def __len__(self):
        return len(self.targets)

if __name__ == '__main__':
    data = Sample(10)
    dataloder = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, num_workers=1)

    for epoch in range(1):
        for i, data in enumerate(dataloder, 0):
            inputs, targets = data
            pdb.set_trace()
            targets = Variable(targets)
