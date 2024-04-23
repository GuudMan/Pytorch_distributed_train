# Pytorch_distributed_train
pytroch分布式训练， DataParallel与DistributedDataParallel， 单机单卡， 多机多卡， cpu训练脚本， 完整的数据与代码

# 数据集：
ChnSentiCorp_htl_all.csv 中文评论数据集

# 分词器
hfl/chinese-macbert-base

# 使用
第一类cpu: pytorch train_cpu.py

第二类DataParallel: python train_multi_gpu_data_parallel.py

第三类DistributedDataParallel: 

第三类有两种启动训练的方式：

运行命令1：

python -m torch.distributed.launch --nproc_per_node=2 train_distributed_data_parallel.py

需要从参数传递中读取local_ran

import argparse

parse = argparse.ArgumentParser()

parse.add_argument('--local-rank', help="local device on current node", type=int)

args = parse.parse_args()

local_rank = args.local_rank


命令2：

torchrun --standalone --nproc_per_node=2 train_distributed_data_parallel.py

需要从环境变量中读取

local_rank = int(os.environ["LOCAL_RANK"])

