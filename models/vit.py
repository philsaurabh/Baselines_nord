from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vision_transformer']


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # Flatten spatial dimensions
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()

        # Linear projections
        q = self.query(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, num_patches, self.embed_dim)
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#class VisionTransformer(nn.Module):
#    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
#        super(VisionTransformer, self).__init__()
#        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#        self.num_patches = self.patch_embed.num_patches
#        self.embed_dim = embed_dim
#
#        # CLS token and positional encoding
#        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
#        self.dropout = nn.Dropout(dropout)
#
#        # Transformer blocks
#        self.blocks = nn.Sequential(
#            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout) for _ in range(depth)]
#        )
#        self.norm = nn.LayerNorm(embed_dim)
#
#        # Classification head
#        self.head = nn.Linear(embed_dim, num_classes)
#
#    def forward(self, x):
#        batch_size = x.size(0)
#        x = self.patch_embed(x)
#
#        # Add CLS token
#        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#        x = torch.cat((cls_tokens, x), dim=1)
#
#        # Add positional encoding
#        x = x + self.pos_embed
#        x = self.dropout(x)
#
#        # Transformer layers
#        x = self.blocks(x)
#
#        # Classification head
#        x = self.norm(x)
#        cls_token_final = x[:, 0]
#        x = self.head(cls_token_final)
#
#        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, is_feat=False, preact=False):
	    batch_size = x.size(0)
	    # Patch Embedding
	    x = self.patch_embed(x)
	    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
	    x = torch.cat((cls_tokens, x), dim=1)
	    x = x + self.pos_embed
	    x = self.dropout(x)
	    f0 = x  # First feature: Embedded patches with positional encoding
	
	    # Transformer blocks
	    f1_pre, f2_pre, f3_pre = None, None, None  # Pre-normalized features
	    for i, block in enumerate(self.blocks):
	        if i == 0:
	            f1_pre = x
	        elif i == len(self.blocks) // 2:
	            f2_pre = x
	        elif i == len(self.blocks) - 1:
	            f3_pre = x
	        x = block(x)
	
	    # Last output after all transformer layers
	    f1, f2, f3 = f1_pre, f2_pre, f3_pre
	    x = self.norm(x)
	    f4 = x[:, 0]  # CLS token as the final feature
	
	    # Classification head
	    logits = self.head(f4)
	
	    if is_feat:
	        if preact:
	            return [f0, f1_pre, f2_pre, f3_pre, f4], logits
	        else:
	            return [f0, f1, f2, f3, f4], logits
	    else:
	        return logits


def vit_tiny(**kwargs):
    return VisionTransformer(embed_dim=64, depth=6, num_heads=4, **kwargs)


def vit_small(**kwargs):
    return VisionTransformer(embed_dim=128, depth=8, num_heads=8, **kwargs)


def vit_base(**kwargs):
    return VisionTransformer(embed_dim=256, depth=12, num_heads=12, **kwargs)


if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = vit_tiny(num_classes=10)
    logits = net(x)
    print(logits.shape)
