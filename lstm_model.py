
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask

def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    NEG_INF = -10000
    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result

class LSTMModel(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, padding_idx,
                 dropout=0.1, bidirectional=True, pretrained_embedding=None, att=True):
        
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.padding_idx = padding_idx

        if pretrained_embedding is not None:
            self.embed = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_embedding).float())
            self.embed.requires_grad_(requires_grad=False)
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        self.fc_att = nn.Linear(hidden_dim * 2, 1)
        
        self.output_layer = nn.Linear(hidden_dim*2, num_classes)
        
    
    def forward(self, x, seq_lens):
        
        max_seq_len = torch.max(seq_lens)
        mask = seq_mask(seq_lens, max_seq_len)
        
        
        embed = self.embed(x)
        #print(embed.shape)
        
        x_packed = pack_padded_sequence(
            embed, seq_lens.to('cpu'), batch_first=True, enforce_sorted=False)
        
        #print(x_packed.shape)
        lstm_packed, _ = self.lstm(x_packed)
        
        lstm_out, length = pad_packed_sequence(lstm_packed, batch_first=True)
        
        att = self.fc_att(lstm_out).squeeze(-1) # [b,msl,h*2]->[b,msl]
        
        att = mask_softmax(att, mask) # [b,msl]
        #print(att.shape)
        #print(lstm_out.shape)
        att_out = torch.sum(att.unsqueeze(-1) * lstm_out, dim=1) # [b,h*2]
        #print(att_out.shape)
        logits = self.output_layer(att_out).squeeze(-1)
        return logits
