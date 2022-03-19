from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, path, sequence):
        self.sequence = sequence
        with open(path, 'r', encoding='utf-8-sig') as f:
            content_list = f.readlines()
        # 归一化
        max_list, min_list = [], []
        data_list = []
        time_list = []
        for index, one_line in enumerate(content_list):
            if index == 0:
                continue
            data_list.append(one_line.replace('\n', '').split(',')[2:])
            time_list.append(one_line.replace('\n', '').split(',')[1])

        data_array = np.array(data_list, dtype=np.float32)
        for i in range(data_array.shape[1]):
            max_list.append(np.max(data_array[:,i]))
            min_list.append(np.min(data_array[:,i]))
            data_array[:,i] = (data_array[:,i] - np.min(data_array[:,i])) / (np.max(data_array[:,i]) - np.min(data_array[:,i]))
        self.max_list, self.min_list = max_list, min_list
        content_list = list(data_array)
        content_list = content_list[0: int(0.7 * len(content_list))]
        time_list = time_list[0:int(0.7 * len(time_list))]
        self.time_list = time_list
        self.content_list = content_list

    def __getitem__(self, item):
        data = self.content_list[item: item + self.sequence]
        label = self.content_list[item + self.sequence][1]
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        return data, label

    def __len__(self):
        return len(self.content_list) - self.sequence


class TestDataset(Dataset):
    def __init__(self, path, sequence):
        self.sequence = sequence
        with open(path, 'r', encoding='utf-8-sig') as f:
            content_list = f.readlines()
        # 归一化
        max_list, min_list = [], []
        data_list = []
        time_list = []
        for index, one_line in enumerate(content_list):
            if index == 0:
                continue
            data_list.append(one_line.replace('\n', '').split(',')[2:])
            time_list.append(one_line.replace('\n', '').split(',')[1])
        data_array = np.array(data_list, dtype=np.float32)
        for i in range(data_array.shape[1]):
            max_list.append(np.max(data_array[:,i]))
            min_list.append(np.min(data_array[:,i]))
            data_array[:,i] = (data_array[:,i] - np.min(data_array[:,i])) / (np.max(data_array[:,i]) - np.min(data_array[:,i]))

        content_list = list(data_array)
        content_list = content_list[int(0.7 * len(content_list)): len(content_list) - 1]
        time_list = time_list[int(0.7 * len(time_list)): len(time_list) - 1]
        time_list = time_list[self.sequence:]
        self.time_list = time_list
        self.content_list = content_list
        self.max_list, self.min_list = max_list, min_list

    def __getitem__(self, item):
        data = self.content_list[item: item + self.sequence]
        label = self.content_list[item + self.sequence][1]

        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        return data, label

    def __len__(self):
        return len(self.content_list) - self.sequence


class TrainDatasetOnlyShoupan(Dataset):
    def __init__(self, path, sequence):
        self.sequence = sequence
        with open(path, 'r', encoding='utf-8-sig') as f:
            content_list = f.readlines()
        # 归一化
        max_list, min_list = [], []
        data_list = []
        time_list = []
        for index, one_line in enumerate(content_list):
            if index == 0:
                continue
            data_list.append(one_line.replace('\n', '').split(',')[2:])
            time_list.append(one_line.replace('\n', '').split(',')[1])

        data_array = np.array(data_list, dtype=np.float32)
        data_array = data_array[:,1]

        self.max_value = np.max(data_array)
        self.min_value = np.min(data_array)
        data_array = (data_array - self.min_value) / (self.max_value - self.min_value)
        content_list = list(data_array)
        content_list = content_list[0: int(0.7 * len(content_list))]
        time_list = time_list[0:int(0.7 * len(time_list))]
        self.time_list = time_list
        self.content_list = content_list

    def __getitem__(self, item):
        data = self.content_list[item: item + self.sequence]
        label = self.content_list[item + self.sequence]
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        return data, label

    def __len__(self):
        return len(self.content_list) - self.sequence

class TestDatasetOnlyShoupan(Dataset):
    def __init__(self, path, sequence):
        self.sequence = sequence
        with open(path, 'r', encoding='utf-8-sig') as f:
            content_list = f.readlines()
        # 归一化
        max_list, min_list = [], []
        data_list = []
        time_list = []
        for index, one_line in enumerate(content_list):
            if index == 0:
                continue
            data_list.append(one_line.replace('\n', '').split(',')[2:])
            time_list.append(one_line.replace('\n', '').split(',')[1])

        data_array = np.array(data_list, dtype=np.float32)
        data_array = data_array[:,1]

        self.max_value = np.max(data_array)
        self.min_value = np.min(data_array)
        data_array = (data_array - self.min_value) / (self.max_value - self.min_value)
        content_list = list(data_array)
        content_list = content_list[int(0.7 * len(content_list)): len(content_list) - 1]
        time_list = time_list[int(0.7 * len(time_list)): len(time_list) - 1]
        self.time_list = time_list
        self.content_list = content_list

    def __getitem__(self, item):
        data = self.content_list[item: item + self.sequence]
        label = self.content_list[item + self.sequence]
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        return data, label

    def __len__(self):
        return len(self.content_list) - self.sequence


if __name__ == '__main__':
    # train_dataset = TrainDataset(r'C:\Users\12517\Desktop\股票分析\上证指数.csv', 30)
    test_dataset = TestDatasetOnlyShoupan(r'C:\Users\12517\Desktop\股票分析\上证指数.csv', 30)
    # train_length = int(0.8 * len(dataset))
    # test_length = len(dataset) - train_length
    # train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_length, test_length])
    train_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print(len(test_dataset.time_list), len(test_dataset.content_list))
    # import torch
    # for data, label in train_loader:
    #     data = torch.unsqueeze(data, dim=2)
    #     print(data.shape)