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

def train(train_data_loader, 
          eval_data_loader, 
          model, 
          optimizer, 
          num_epoch, 
          log_step_interval, 
          save_step_interval, 
          eval_step_interval, 
          save_path, 
          resume=""):
    start_epoch = 0
    start_step = 0
    if resume != "":
        # 加载之前训练过得模型的参数文件
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
    
    for epoch_index in range(start_epoch, num_epoch):
        ema_loss = 0.
        num_batches = len(train_data_loader)

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches * (epoch_index) + batch_index + 1
            # print('-----------------')
            # print(token_index.shape)
            # exit(-1)
            logits = model(token_index)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(logits), 
                                              F.one_hot(target, num_classes=2).to(torch.float32))
            ema_loss = 0.9 * ema_loss + 0.1 * bce_loss
            bce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if step % log_step_interval == 0:
                logging.warning(f"epoch_index:{epoch_index}, batch_index:{batch_index}, ema_loss:{ema_loss.item()}")
            
            if step % save_step_interval == 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({
                    'epoch': epoch_index, 
                    'step': step, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': bce_loss, 
                }, save_file)
                logging.warning(f"checkpoint has been save in {save_file}")

            if step % eval_step_interval == 0:
                logging.warning("start to do evaluation...")
                model.eval()
                ema_eval_loss = 0
                total_acc_account = 0
                total_account = 0

                for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
                    total_account += eval_target.shape[0]
                    eval_logits = model(eval_token_index)
                    total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), 
                                                           F.one_hot(eval_target, num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss
                
                acc = total_acc_account / total_account

                logging.warning(f"eval_ema_loss:{ema_eval_loss.item()}, eval_acc:{acc}")
                model.train()
                                


if __name__ == '__main__':
    vocab_size = tokenizer.vocab_size
    emb_dim = 128
    model = MyModel(vocab_size, emb_dim, 2)
    print("模型总参数：", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    resume = ""
    train(train_data_loader, 
          val_data_loader, 
          model, 
          optimizer, 
          num_epoch=1, 
          log_step_interval=20, 
          save_step_interval=500, 
          eval_step_interval=300, 
          save_path="./logs_review_text_classification",
          resume=resume
          )








