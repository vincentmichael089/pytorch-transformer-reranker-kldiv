# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import math

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class PositionalEncoding(pl.LightningModule):
  def __init__(self, num_hiddens, dropout, max_len=2000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("P", torch.zeros((1, max_len, num_hiddens)))
    self.register_buffer("tempX", torch.arange(max_len, dtype=torch.float32, device = self.device).reshape(-1, 1) / 
      torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32, device = self.device) / num_hiddens))
    self.P[:, :, 0::2] = torch.sin(self.tempX)
    self.P[:, :, 1::2] = torch.cos(self.tempX)

  def forward(self, X):
    X = X + self.P[:, :X.shape[1], :]
    return self.dropout(X)

class SegmentEncoding(pl.LightningModule):
  def __init__(self, vocab_size: int, emb_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens):
    return self.embedding(tokens.to(self.device)) * math.sqrt(self.emb_size)

class TokenEmbedding(pl.LightningModule):
  def __init__(self, vocab_size: int, emb_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens):
    return self.embedding(tokens.to(self.device)) * math.sqrt(self.emb_size)

# modified from fairseq's RobertaClassificationHead
class ClassificationHead(pl.LightningModule):
  def __init__(
    self,
    input_dim, inner_dim,
    num_classes,
    dropout
  ):
    super().__init__()
    self.dense = nn.Linear(input_dim, inner_dim)
    self.dropout = nn.Dropout(dropout)
    self.activation_fn = nn.Tanh()
    self.sigmoid = nn.Sigmoid()
    self.output_projection = nn.Linear(inner_dim, num_classes)

  def forward(self, features): # B x L x D
    x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = self.activation_fn(x)
    x = self.dropout(x)
    return self.output_projection(x) # B x 1

class VanillaTransformerEncoder(pl.LightningModule):
  def  __init__(
    self,
    num_encoder_layers,
    emb_size,
    nhead,
    src_vocab_size,
    dim_ff,
    dropout
  ):
    super().__init__()
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=emb_size, 
      nhead=nhead, 
      dim_feedforward=dim_ff, 
      dropout=dropout, 
      activation=nn.functional.relu, 
      layer_norm_eps= 1e-5, 
      batch_first=True, 
      norm_first=False)

    encoder_norm = nn.LayerNorm(emb_size, 1e-5)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
  
  def forward(self, x, padding_mask):
    return self.encoder(src=x, src_key_padding_mask=padding_mask)

class TransformerReranker(pl.LightningModule):
  def __init__(
    self,
    num_encoder_layers,
    emb_size,
    nhead,
    vocab_size,
    dim_ff,
    dropout
  ):
    super().__init__()
    self.positional_encoding = PositionalEncoding(emb_size, dropout = dropout)
    self.segment_encoding = SegmentEncoding(3, emb_size)
    self.tok_emb = TokenEmbedding(vocab_size, emb_size)
    self.encoder = VanillaTransformerEncoder(num_encoder_layers, emb_size, nhead, vocab_size, dim_ff, dropout)  
    self.classifier = ClassificationHead(emb_size, emb_size, 1, dropout)

  def forward(self, input, segment, padding_mask):
    tok = self.tok_emb(input)
    seg = self.segment_encoding(segment)
    tok = self.positional_encoding(tok + seg)
    encoded = self.encoder(tok, padding_mask)
    return self.classifier(encoded)

class PlTransformer(pl.LightningModule):
  def __init__(self, model, lr):
    super().__init__()
    self.model = model
    self.lr = lr
    self.kld = torch.nn.KLDivLoss(reduction="sum", log_target=False)
    self.EPSILON = torch.finfo(torch.float32).eps
    self.TEMPERATURE = 1 
    self.softmax = nn.Softmax(dim=-2)

    for p in self.model.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  
  def training_step(self, batch, batch_idx): 
    input = batch[0]
    segment = batch[1]
    bleu = batch[2]
    padding_mask =  (input == PAD_IDX)
 
    # predicted distribution
    logits = self.model(input, segment, padding_mask)
    model_dist = torch.nn.functional.log_softmax(logits.t(), dim=1, dtype=torch.float32).squeeze()

    # target distribution
    bleu = bleu.unsqueeze(0)
    min_v = torch.min(bleu, 1, keepdim=True).values
    max_v = torch.max(bleu, 1, keepdim=True).values
    norm_target = (bleu - min_v) / (max_v - min_v + self.EPSILON)
    target_dist = torch.nn.functional.softmax(norm_target, dim=-1).squeeze()

    loss = self.kld(model_dist, target_dist)
    self.log('training_loss', loss, on_step=True, on_epoch=True)
    return loss

  def validation_step(self, batch, batch_idx):
    input = batch[0]
    segment = batch[1]
    bleu = batch[2]
    padding_mask =  (input == PAD_IDX)
 
    # predicted distribution
    logits = self.model(input, segment, padding_mask)
    model_dist = torch.nn.functional.log_softmax(logits.t(), dim=1, dtype=torch.float32).squeeze()

    # target distribution
    bleu = bleu.unsqueeze(0)
    min_v = torch.min(bleu, 1, keepdim=True).values
    max_v = torch.max(bleu, 1, keepdim=True).values
    norm_target = (bleu - min_v) / (max_v - min_v + self.EPSILON)
    target_dist = torch.nn.functional.softmax(norm_target, dim=-1).squeeze()

    loss = self.kld(model_dist, target_dist)
    self.log('validation_loss', loss, on_step=True, on_epoch=True)
    return loss


  def predict_step(self, batch, batch_idx):
    input = batch[0]
    segment = batch[1]
    padding_mask =  (input == PAD_IDX)

    logits = self.model(input, segment, padding_mask)
    softmax_logits = self.softmax(logits)
    position = softmax_logits.argmax().item()
    return input[position]


  def configure_optimizers(self):
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)    
    scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=10000,
      num_training_steps= 235200 
      )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, 'monitor':"val_loss"}
    return [optimizer], [scheduler]


