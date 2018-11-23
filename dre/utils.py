import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from sklearn import preprocessing

class SubsetSequentialSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def csv_load(filename):#$, window=6, jump=1):
    df = pd.read_csv(filename)
    label = df['changepoint'].values
    df = df.iloc[:, 2:]
    df = df.fillna(method='backfill')

    min_max_scaler = preprocessing.MinMaxScaler()
    nm_scaled = min_max_scaler.fit_transform(df)

    #std_scaler = preprocessing.StandardScaler()
    #std_scaled = std_scaler.fit_transform(df)

    values = nm_scaled #std_scaled #df.values
    return values.astype(np.float32), label

class IoTCNNDataset(Dataset):
    def __init__(self, csv_file, window=6, jump=1, n=50, L=None):
        self.features, self.labels = csv_load(csv_file)
        if L is not None:
            self.features, self.labels = self.features[:L], self.labels[:L]
        print(len(self.features))
        print(self.labels.sum())
        input()
        self.window = window
        self.jump = jump
        self.n = n
        self.shape = (window, self.features.shape[-1])

    def __len__(self):
        L = len(self.features)
        #return int((L - (self.window * self.n * 2 - 1) - 1) / self.jump + 1)# - (2 * self.n - 1)
        return int((L - (self.window + self.jump * (self.n * 2 - 1) - 1) - 1) / self.jump + 1)
        # is it right??

    def __getitem__(self, idx):
        # TODO numpy to tensor data
        input_dim = self.features.shape[-1]
        #print("data shape", self.features.shape, len(self))
        #print("idx", idx, idx * self.jump, idx*self.jump + self.window * self.n * 2)
        features = self.features[idx*self.jump:
                (idx + 2*self.n - 1)*self.jump + self.window]# * self.jump * self.n * 2].reshape(self.n * 2, self.window, -1)

        X_ref = np.zeros((self.n, self.window, features.shape[-1]),
                dtype=np.float32)
        X_test = np.zeros((self.n, self.window, features.shape[-1]),
                dtype=np.float32)

        for i in range(self.n):
            # self.window x input_dim
            X_ref[i] = features[self.jump*i: self.jump*i + self.window]
            X_test[i] = features[self.jump*(i+self.n): self.jump*(i+self.n) +
                    self.window]

        #label = self.labels[(idx + self.n)*self.jump:
        #        (idx + self.n)*self.jump + self.window]
        label = self.labels[(idx + self.n)*self.jump:
                (idx + self.n)*self.jump + self.jump]
        label = 1 if np.any(label == 1) else 0
        #if idx > 0:
        #    prev_label = self.labels[(idx-1 + self.n) * self.jump:
        #            (idx - 1 + self.n)*self.jump + self.window]
            #prev_label = self.labels[(idx - 1) * self.jump + self.window * self.n
            #        :(idx-1)*self.jump + self.window * self.n + self.window]
        #    prev_label = 1 if np.any(prev_label == 1) else 0
        #    if prev_label == 1:
        #        label = 0
        #sample = {'features': features, 'label': label}
        return X_ref, X_test, label



def data_load(filename='../N1Lounge8F_06/n1lounge8f_06_10sec.csv',
               validation_split=0.5,
               window=6, jump=1, batch_size=100, n=50, L=None):
    print("L", L)
    print("n", n)
    n1lounge = IoTCNNDataset(filename, window=window, jump=jump, n=n, L=L)
    # train, val split
    n_data = len(n1lounge)
    indices = np.array(range(n_data))

    split = int(np.floor(validation_split * n_data))
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    train_sampler_seq = SubsetSequentialSampler(train_indices)
    val_sampler = SubsetSequentialSampler(val_indices)
    train_loader = DataLoader(n1lounge, batch_size=batch_size, num_workers=10,
            sampler=train_sampler)
    train_loader_seq = DataLoader(n1lounge, batch_size=batch_size, num_workers=10,
            sampler=train_sampler_seq)
    val_loader = DataLoader(n1lounge, batch_size=batch_size, num_workers=10,
            sampler=val_sampler)
    return train_loader, val_loader, train_loader_seq, n1lounge.shape[0], n1lounge.shape[1]


if __name__ == '__main__':

    train_loader, val_loader = data_load(window=30, jump=6)

    for i_batch, (features, label) in enumerate(train_loader):
        print(i_batch, features.size(),
           label.size()
           )
        if i_batch == 3:
            break
