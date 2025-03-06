import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[3].weight)

    def forward(self, x):
        return self.net(x)

class MLA(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent
        
        # 添加输入归一化
        self.input_norm = nn.LayerNorm(d_model)
        
        # 使用更稳定的初始化
        self.q_proj = nn.Linear(d_model, d_latent * num_heads)
        self.k_proj = nn.Linear(d_model, d_latent * num_heads)
        self.v_proj = nn.Linear(d_model, d_latent * num_heads)
        self.o_proj = nn.Linear(d_latent * num_heads, d_model)
        
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        
    def _reset_parameters(self):
        # 使用更保守的初始化
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def forward(self, x, context=None):
        # 添加输入检查
        if torch.isnan(x).any():
            print("Input contains NaN in MLA")
        
        # 如果没有提供context，就使用x作为context（自注意力）
        if context is None:
            context = x
        
        # 添加输入归一化
        x = self.input_norm(x)
        
        batch, seq, _ = x.size()
        
        # 投影并reshape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_latent)
        k = self.k_proj(context).view(batch, -1, self.num_heads, self.d_latent)
        v = self.v_proj(context).view(batch, -1, self.num_heads, self.d_latent)
        
        # 转置以进行注意力计算
        q = q.transpose(1, 2)  # (batch, num_heads, seq, d_latent)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_latent)
        
        # 使用更稳定的softmax
        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)  # (batch, num_heads, seq, d_latent)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        
        # 输出投影
        out = self.o_proj(out)
        
        return out

class TransformerBlockMLA(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1):
        super().__init__()
        self.attn = MLA(d_model, num_heads, d_latent, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1):
        super().__init__()
        self.self_attn = MLA(d_model, num_heads, d_latent, dropout=dropout)
        self.cross_attn = MLA(d_model, num_heads, d_latent, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(x + self.cross_attn(x, encoder_output))
        x = self.ln3(x + self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_latent, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlockMLA(d_model, num_heads, d_latent, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_latent, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_latent, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output):
        for block in self.blocks:
            x = block(x, encoder_output)
        return x

class DeepCADMLA(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_latent, dropout=0.1):
        super().__init__()
        
        # 添加嵌入层归一化
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        
        # 编码器和解码器
        self.encoder = Encoder(d_model, num_heads, num_layers, d_latent, dropout)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_latent, dropout)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        # 初始化
        self._init_parameters()
    
    def _init_parameters(self):
        # 使用更保守的初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        # 输出层初始化
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_seq, target_seq):
        # 添加输入检查
        if torch.isnan(input_seq).any() or torch.isnan(target_seq).any():
            print("Input contains NaN in DeepCADMLA")
        
        # 编码
        encoder_output = self.encoder(input_seq)
        
        # 解码
        decoder_output = self.decoder(target_seq, encoder_output)
        
        # 输出层
        output = self.output_layer(decoder_output)
        
        return output

# 实验组4：标准注意力
class StandardAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model 必须能被 num_heads 整除"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
            
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]
        
        # 投影和重塑
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重新形状和投影
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out

# 实验组5：局部注意力
class LocalAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        
        assert self.head_dim * num_heads == d_model, "d_model 必须能被 num_heads 整除"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
            
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]
        
        # 投影和重塑
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 创建局部注意力掩码
        local_mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(context_len, i + self.window_size // 2 + 1)
            local_mask[:, :, i, start:end] = False
            
        # 应用局部掩码
        scores = scores.masked_fill(local_mask, -1e9)
        
        # 应用额外掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重新形状和投影
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out

# 实验组6：稀疏注意力
class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity=0.7, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sparsity = sparsity
        
        assert self.head_dim * num_heads == d_model, "d_model 必须能被 num_heads 整除"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
    
    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
            
        batch_size, seq_len, _ = x.shape
        context_len = context.shape[1]
        
        # 投影和重塑
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(batch_size, context_len, self.num_heads, self.head_dim)
        
        # 转置为 [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 创建稀疏掩码：保留每行最大的topk个值
        if self.training:
            k = int((1 - self.sparsity) * context_len)
            if k < 1:
                k = 1
                
            # 获取每行最大的topk个值的索引
            topk_indices = torch.topk(scores, k, dim=-1)[1]
            
            # 创建稀疏掩码
            sparse_mask = torch.ones_like(scores, dtype=torch.bool)
            
            # 根据topk索引填充掩码
            for b in range(batch_size):
                for h in range(self.num_heads):
                    for i in range(seq_len):
                        sparse_mask[b, h, i, topk_indices[b, h, i]] = False
                        
            # 应用稀疏掩码
            scores = scores.masked_fill(sparse_mask, -1e9)
            
        # 应用额外掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 重新形状和投影
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out