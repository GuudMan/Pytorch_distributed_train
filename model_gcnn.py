import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_class):
        super(MyModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_class = num_class

        self.embedding_table = nn.Embedding(self.vocab_size, embedding_dim=self.emb_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(self.emb_dim, 64, 15, stride=7)
        self.conv_B_1 = nn.Conv1d(self.emb_dim, 64, 15, stride=7)

        self.conv_A_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, self.num_class)
    
    def forward(self, word_index):
        # 通过word_index得到word_embedding
        # [bs, max_seq_len, embedding_dim]
        word_embedding = self.embedding_table(word_index)  
        
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B)  # [bs, 64, max_seq_len]


        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)  # [bs, 64, max_seq_len]


        # 3. 池化并经过全连接层
        pool_output = torch.mean(H, dim=-1)  # 平均池化， 得到[bs, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output)
        return logits
    