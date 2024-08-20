import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, labels, smooth_eps=0.1, mask=None, from_logits=True):
        """
        Args:
            logits: (N, Lv), unnormalized probabilities, torch.float32
            labels: (N, Lv) or (N, ), one hot labels or indices labels, torch.float32 or torch.int64
            smooth_eps: float
            mask: (N, Lv)
            from_logits: bool
        """
        if from_logits:
            probs = F.log_softmax(logits, dim=-1)
        else:
            probs = logits
        num_classes = probs.size()[-1]
        if len(probs.size()) > len(labels.size()):
            labels = onehot(labels, num_classes).type(probs.dtype)
        if mask is None:
            labels = labels * (1 - smooth_eps) + smooth_eps / num_classes
        else:
            mask = mask.type(probs.dtype)
            valid_samples = torch.sum(mask, dim=-1, keepdim=True, dtype=probs.dtype)  # (N, 1)
            eps_per_sample = smooth_eps / valid_samples
            labels = (labels * (1 - smooth_eps) + eps_per_sample) * mask
        loss = -torch.sum(labels * probs, dim=-1)
        if self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss  # (N, )


class MILNCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MILNCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]
        x = x.view(bsz, bsz, -1) # (N, N, 1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1) # (N, 1)
        nominator = torch.logsumexp(nominator, dim=1) # (N, )  
        
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1) # bs,2bs
        denominator = torch.logsumexp(denominator, dim=1) # (N,)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input_tensor = torch.randn(32, 300, 20)
        >>> output = m(input_tensor)
    """
    def __init__(self, in_ch, out_ch, k, dim=1, relu=True):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        """
        super(DepthwiseSeparableConv, self).__init__()
        self.relu = relu
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (N, L_in, D)
        :Output: (N, L_out, D)
        """
        x = x.transpose(1, 2)
        if self.relu:
            out = F.relu(self.pointwise_conv(self.depthwise_conv(x)), inplace=True)
        else:
            out = self.pointwise_conv(self.depthwise_conv(x))
        return out.transpose(1, 2)  # (N, L, D)


        

class ConvEncoder(nn.Module):
    def __init__(self, kernel_size=7, n_filters=128, dropout=0.1):
        super(ConvEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_filters)
        self.conv = DepthwiseSeparableConv(in_ch=n_filters, out_ch=n_filters, k=kernel_size, relu=True)

    def forward(self, x):
        """
        :param x: (N, L, D)
        :return: (N, L, D)
        """
        return self.layer_norm(self.dropout(self.conv(x)) + x)  # (N, L, D)


class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
       
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids) # (N, L, D)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertLayer(nn.Module):
    def __init__(self, config, use_self_attention=True):
        super(BertLayer, self).__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states:  (N, L, D)
            attention_mask:  (N, L) with 1 indicate valid, 0 indicates invalid
        """
        if self.use_self_attention:
            attention_output = self.attention(hidden_states, attention_mask)
        else:
            attention_output = hidden_states
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
            attention_mask: (N, Lq, L) ?
        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask) #(N, L, D)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.ReLU(True))

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask,
                RETURN_PROBS:bool=False):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, 1, L)
        """
        attention_mask = (1 - attention_mask.unsqueeze(1)) * - 10000.  # (N, 1, 1, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose for mutihhead attention
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer) # (N, nh, Lq, dh)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (N, Lq, nh, dh)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer #(N, Lq, D) input: (N, Lq, D)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states) # (N, L, D)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class UnifiedQueryEncoder(nn.Module):
    def __init__(self, config,SEPARATE=False,QUERY_INPUT_SIZE=None,N_LAYERS=1,N_HEADS=None):
        super(UnifiedQueryEncoder, self).__init__()
        self.config = config
        self.SEPARATE = SEPARATE
        self.hidden_size = config.hidden_size
        if(QUERY_INPUT_SIZE is not None):
            self.mutimodal_latent_proj = nn.Linear(QUERY_INPUT_SIZE, self.hidden_size)
        else:
            self.mutimodal_latent_proj = nn.Linear(config.query_input_size, self.hidden_size)
        self.mutimodal_query_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_mutimodal_l,
            hidden_size=self.hidden_size, 
            dropout=config.input_drop
        )
        self.mutimodal_query_encoder_layer_config = edict(
            hidden_size=self.hidden_size, 
            intermediate_size=self.hidden_size,                                  
            hidden_dropout_prob=config.drop, 
            num_attention_heads=N_HEADS if N_HEADS is not None else config.n_heads,
            attention_probs_dropout_prob=config.drop
        )
        self.mutimodal_query_encoder =  BertAttention(self.mutimodal_query_encoder_layer_config)
        
    def forward(self,desc_query,desc_query_mask,image_query,image_query_mask):
        '''
        Args:
            desc_query: (N, Lq, Dq)
            desc_query_mask: (N, Lq)
            image_query: (N, Li, Di)
            image_query_mask: (N, Li)
            
            output: (N, Lq+Li, hidden_size)
        '''
        
        
        mutimodal_query = torch.cat((image_query,desc_query),dim=1) # (N, Lq+Li, D)
        mutimodal_query_mask = torch.cat((image_query_mask,desc_query_mask),dim=1) # (N, Lq+Li)

        _mutimodal_query_mask = mutimodal_query_mask.unsqueeze(1) # (N, 1, Lq+Li)
        mutimodal_query = self.mutimodal_latent_proj(mutimodal_query)
        mutimodal_query = self.mutimodal_query_pos_embed(mutimodal_query)
        mutimodal_query = self.mutimodal_query_encoder(
            mutimodal_query,
            _mutimodal_query_mask
        )
        
        desc_query_mask = torch.ones_like(desc_query_mask,dtype=desc_query_mask.dtype)
        image_query_mask = torch.ones_like(image_query_mask,dtype=image_query_mask.dtype)
        if(self.SEPARATE):
            desc_query_output,image_query_output = mutimodal_query.split([desc_query.size(1),image_query.size(1)],dim=1)
            return image_query_output,image_query_mask,desc_query_output,desc_query_mask
        else:
            mutimodal_query_mask = torch.cat((image_query_mask,desc_query_mask),dim=1) # (N, Lq+Li)
            return mutimodal_query,mutimodal_query_mask

class EMP_Layers(nn.Module):
    def __init__(self,config,N_LAYERS=1):
        super(EMP_Layers, self).__init__()      
        self.config = config
        self.N_LAYERS = N_LAYERS
        
        self.Input_UnifiedQueryEncoder = UnifiedQueryEncoder(config,SEPARATE=True)
        self.UnifiedQueryEncoders = nn.ModuleList([UnifiedQueryEncoder(config,SEPARATE=True,QUERY_INPUT_SIZE=config.hidden_size) for _ in range(N_LAYERS)])
    
    def forward(self,desc_query,desc_query_mask,image_query,image_query_mask):
        '''
        Args:
            desc_query: (N, Lq, Dq)
            desc_query_mask: (N, Lq)
            image_query: (N, Li, Di)
            image_query_mask: (N, Li)
        '''
       
        _image_query,_image_query_mask,_desc_query,_desc_query_mask = \
        self.Input_UnifiedQueryEncoder(desc_query,desc_query_mask,image_query,image_query_mask)
        for i in range(self.N_LAYERS):
            _image_query,_image_query_mask,_desc_query,_desc_query_mask = \
            self.UnifiedQueryEncoders[i](_desc_query,_desc_query_mask,_image_query,_image_query_mask)
        
        mutimodal_query = torch.cat((_image_query,_desc_query),dim=1)
        mutimodal_query_mask = torch.cat((_image_query_mask,_desc_query_mask),dim=1)
        return mutimodal_query,mutimodal_query_mask