import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer



class textDataset(Dataset):
    def __init__(self,csv_path):
        super().__init__()
        self.install_data = pd.read_csv(csv_path, encoding='utf8', sep=',', header=0)
        self.install_data = self.install_data.dropna(how='any')
        
    def __len__(self):
        return len(self.install_data)
    
    def __getitem__(self, idx):
        
        input_data = pd.DataFrame([])
        label = pd.DataFrame([])
        
        input_data = self.install_data.iloc[idx, 1]
        
        # input_ids = tokenizer.encode(input_data, padding="max_length", max_length=120, truncation=True)
        label = self.install_data.iloc[idx, 0]
        return label, input_data


if __name__ == '__main__':
    tD = textDataset('./ChnSentiCorp_htl_all.csv')
    for i, (label, data) in enumerate(tD):
        print(label)
        print(data)
        if i == 5:
            break
    # print(tokenizer.vocab_size)