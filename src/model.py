#定义新的MLA模型
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MLA(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, is_cross=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.d_latent = d_latent
        self.is_cross = is_cross

        if is_cross:
            self.w_latent_kv = nn.Linear(d_model, d_latent)
            self.w_k_heads = nn.ModuleList([nn.Linear(d_latent, self.d_h) for _ in range(num_heads)])
            self.w_v_heads = nn.ModuleList([nn.Linear(d_latent, self.d_h) for _ in range(num_heads)])
            self.w_q = nn.Linear(d_model, d_model)
        else:
            self.w_latent = nn.Linear(d_model, d_latent)
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k_heads = nn.ModuleList([nn.Linear(d_latent, self.d_h) for _ in range(num_heads)])
            self.w_v_heads = nn.ModuleList([nn.Linear(d_latent, self.d_h) for _ in range(num_heads)])

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None):
        if not self.is_cross:
            assert key is None and value is None
            x = query
            batch, seq, d = x.size()
            assert d == self.d_model
            z = self.w_latent(x)
            q = self.w_q(x).view(batch, seq, self.num_heads, self.d_h).transpose(1, 2)
            k_heads = []
            v_heads = []
            for i in range(self.num_heads):
                k_i = self.w_k_heads[i](z)
                v_i = self.w_v_heads[i](z)
                k_heads.append(k_i)
                v_heads.append(v_i)
            k = torch.stack(k_heads, dim=1)
            v = torch.stack(v_heads, dim=1)
        else:
            assert key is not None and value is not None
            batch, seq_q, d = query.size()
            batch, seq_k, d = key.size()
            assert d == self.d_model
            z_kv = self.w_latent_kv(key)
            k_heads = []
            v_heads = []
            for i in range(self.num_heads):
                k_i = self.w_k_heads[i](z_kv)
                v_i = self.w_v_heads[i](z_kv)
                k_heads.append(k_i)
                v_heads.append(v_i)
            k = torch.stack(k_heads, dim=1)
            v = torch.stack(v_heads, dim=1)
            q = self.w_q(query).view(batch, seq_q, self.num_heads, self.d_h).transpose(1, 2)

        attn = torch.einsum('bhqd, bhkd -> bhqk', q, k) / (self.d_h ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        output = torch.einsum('bhqk, bhkd -> bhqd', attn, v).transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        output = self.w_o(output)
        return output


class TransformerBlockMLA(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        super().__init__()
        self.attn = MLA(d_model, num_heads, d_latent, is_cross=False)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        super().__init__()
        self.self_attn = MLA(d_model, num_heads, d_latent, is_cross=False)
        self.cross_attn = MLA(d_model, num_heads, d_latent, is_cross=True)
        self.ffn = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(x + self.cross_attn(x, encoder_output, encoder_output))
        x = self.ln3(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_latent):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlockMLA(d_model, num_heads, d_latent) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_latent):
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(d_model, num_heads, d_latent) for _ in range(num_layers)])

    def forward(self, x, encoder_output):
        for block in self.blocks:
            x = block(x, encoder_output)
        return x


class DeepCADMLA(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_latent):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(d_model, num_heads, num_layers, d_latent)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_latent)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_seq, target_seq):
        encoded = self.embedding(input_seq)
        encoder_output = self.encoder(encoded)
        decoder_input = self.embedding(target_seq[:, :-1])  # 教师强制，移除最后一个标记
        decoder_output = self.decoder(decoder_input, encoder_output)
        output = self.output_layer(decoder_output)
        return output