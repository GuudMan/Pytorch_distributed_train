import torch
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
from model_gcnn import MyModel
from load_data import textDataset
from transformers import AutoTokenizer
import sys
import os
import logging
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

BATCH_SIZE = 4


tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
tokenizer.save_pretrained("./chinese_macbert_base")

def collate_fn(batch):
    target = []
    token_index = []
    max_length = emb_dim
    for i, (label, comment) in enumerate(batch):
        input_ids = tokenizer.encode(comment, 
                                     truncation=True, 
                                     add_special_tokens=True, 
                                     padding="max_length", 
                                     max_length=max_length)
        token_index.append(input_ids)
        target.append(label)
    return torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32)


dataset = textDataset(csv_path = "./ChnSentiCorp_htl_all.csv")
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


# collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch.utils.data import DataLoader
train_data_loader = DataLoader(train_dataset, 
                batch_size=BATCH_SIZE, 
                collate_fn=collate_fn, 
                shuffle=True)

val_data_loader = DataLoader(val_dataset, 
                             batch_size=BATCH_SIZE, 
                             collate_fn=collate_fn, 
                             shuffle=False)


                                


if __name__ == '__main__':
    vocab_size = tokenizer.vocab_size
    emb_dim = 128
    model = MyModel(vocab_size, emb_dim, 2)
    # print("模型总参数：", sum(p.numel() for p in model.parameters()))
    # model = torch.load(model.load_state_dict('./param/step_1500.pt'))
    model.load_state_dict(torch.load('./param/step_1500.pt')['model_state_dict'])

    sentence = "位置比较好，但条件一般，居然空调是单冷的，前台服务非常冷淡"
    


    input_ids = tokenizer.encode(sentence, padding="max_length", max_length=128, truncation=True)
    with torch.no_grad():
        input_data = torch.tensor([input_ids])
        output = model(input_data)
        print(output)
         # 提取输出类别
        _, predicted_class = output.max(1)
        print(predicted_class)





