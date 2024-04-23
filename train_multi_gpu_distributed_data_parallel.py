import torch
import torch.distributed
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
from model_gcnn import MyModel
from load_data import textDataset
from transformers import AutoTokenizer
import sys
import os
import logging

"""
运行命令1：
python -m torch.distributed.launch --nproc_per_node=2 train_distributed_data_parallel.py
# 需要从参数传递中读取local_ran
# import argparse
# parse = argparse.ArgumentParser()
# parse.add_argument('--local-rank', help="local device on current node", type=int)
# args = parse.parse_args()
# local_rank = args.local_rank

命令2：
torchrun --standalone --nproc_per_node=2 train_distributed_data_parallel.py
# 需要从环境变量中读取
local_rank = int(os.environ["LOCAL_RANK"])
"""

# os.environ['LOCAL_RANK'] = '0,1'
local_rank = int(os.environ["LOCAL_RANK"])

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

# train_sampler = torch.utils.data.distributed.
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)


# collator = DataCollatorWithPadding(tokenizer=tokenizer)



def train(train_dataset, 
          val_dataset, 
          model, 
          optimizer, 
          num_epoch, 
          log_step_interval, 
          save_step_interval, 
          eval_step_interval, 
          save_path, 
          local_rank,
          resume="", 
          ):
    
    from torch.utils.data import DataLoader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, 
                    batch_size=BATCH_SIZE, 
                    collate_fn=collate_fn, 
                    sampler=train_sampler)
    
    val_data_loader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, 
                                collate_fn=collate_fn, 
                                shuffle=False)
    
    start_epoch = 0
    start_step = 0
    if resume != "":
        # 加载之前训练过得模型的参数文件
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume, map_location=torch.device("cuda:0")) #可以是cpu,cuda,cuda:index
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
    
    for epoch_index in range(start_epoch, num_epoch):
        ema_loss = 0.

        num_batches = len(train_data_loader)

        train_sampler.set_epoch(epoch_index) # 为了让每张卡在每个周期中得到的数据是随机的

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches * (epoch_index) + batch_index + 1

            # 数据拷贝
            target = target.cuda(local_rank)
            token_index = token_index.cuda(local_rank)

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

                for eval_batch_index, (eval_target, eval_token_index) in enumerate(val_data_loader):
                    
                    total_account += eval_target.shape[0]

                                # 数据拷贝
                    eval_target = eval_target.cuda(local_rank)
                    eval_token_index = eval_token_index.cuda(local_rank)

                    eval_logits = model(eval_token_index)
                    total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), 
                                                           F.one_hot(eval_target, num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss
                
                acc = total_acc_account / total_account

                logging.warning(f"eval_ema_loss:{ema_eval_loss.item()}, eval_acc:{acc}")
                model.train()
                                


if __name__ == '__main__':
    import time
    start_t = time.time()
    # import argparse
    # parse = argparse.ArgumentParser()
    # parse.add_argument('--local-rank', help="local device on current node", type=int)
    # args = parse.parse_args()
    # local_rank = args.local_rank

    # n_gpus = 2
    # torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    # torch.cuda.set_device(args.local_rank)
    n_gpus = 2
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)

    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    vocab_size = tokenizer.vocab_size
    emb_dim = 128
    model = MyModel(vocab_size, emb_dim, 2)

    # 模型拷贝， 放入DistributedDataParallel中
    model = nn.parallel.DistributedDataParallel(model.cuda(local_rank),device_ids=[local_rank])
    print("模型总参数：", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    resume = ""
    train(train_dataset, 
          val_dataset, 
          model, 
          optimizer, 
          num_epoch=1, 
          log_step_interval=20, 
          save_step_interval=500, 
          eval_step_interval=300, 
          save_path="./logs_review_text_classification",
          local_rank=local_rank,
          resume=resume
          )
    
    end_t = time.time()
    print("Time Cost: ", round(end_t - start_t))








