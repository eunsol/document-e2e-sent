import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from random import shuffle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import graph_results as plotter
import data_processor as parser
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

num_mentions_map = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3,
                    16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 4}
# (prev #, next #]
NUM_MENTIONS_CUTOFF = [0, 1, 5, 10, 20]
num_mentions_cats = 5

'''
#  in1_features: dimension of in1 embeds
#  in2_Features: dimension of in2 embeds
#  out_features: 1 output per label
m = nn.Bilinear(in1_features=20, in2_features=20, out_features=3)
input1 = torch.randn(128, 10, 20)
input2 = torch.randn(128, 10, 20)
output = m(input1, input2)
print(output.size())  # 128, 10, 40
'''


def aggregate_mentions(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class Model1(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size,
                 hidden_dim, word_embeddings, num_polarities, batch_size,
                 dropout_rate, max_co_occurs):
        super(Model1, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Specify embedding layers
        self.word_embeds = nn.Embedding(vocab_size, embeddings_size)
        self.word_embeds.weight.data.copy_(torch.FloatTensor(word_embeddings))
        # self.word_embeds.weight.requires_grad = False  # don't update the embeddings
        self.polarity_embeds = nn.Embedding(num_polarities + 1, embeddings_size)  # add 1 for <pad>
        self.co_occur_embeds = nn.Embedding(max_co_occurs, embeddings_size)
        self.holder_target_embeds = nn.Embedding(5, embeddings_size)  # add 2 for <pad> and <unk>
        self.num_holder_mention_embeds = nn.Embedding(num_mentions_cats, embeddings_size)
        self.num_target_mention_embeds = nn.Embedding(num_mentions_cats, embeddings_size)
        self.min_mention_embeds = nn.Embedding(num_mentions_cats, embeddings_size)
        self.holder_rank_embeds = nn.Embedding(5, embeddings_size)
        self.target_rank_embeds = nn.Embedding(5, embeddings_size)

        # The LSTM takes [word embeddings, feature embeddings, holder/target embeddings] as inputs, and
        # outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(2 * embeddings_size, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # The linear layer that maps from hidden state space to target space
        self.hidden2label = nn.Linear(2 * hidden_dim, num_labels)

        # Matrix of weights for each layer
        # Linear map from hidden layers to alpha for that layer
        # self.attention = nn.Linear(2 * hidden_dim, 1)
        # Attempting feedforward attention, using 2 layers and sigmoid activation fxn
        # Last layer acts as the w_alpha layer
        self.attention = FeedForward(input_dim=2 * hidden_dim, num_layers=2, hidden_dims=[hidden_dim, 1],
                                     activations=nn.Sigmoid())

        # Span embeddings
        self._endpoint_span_extractor = EndpointSpanExtractor(2 * hidden_dim,
                                                              combination="x,y")
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=2 * hidden_dim)

        # FFNN for holder/target spans respectively
        self.holder_FFNN = FeedForward(input_dim=3 * 2 * hidden_dim, num_layers=2, hidden_dims=[hidden_dim, 1],
                                       activations=nn.ReLU())
        self.target_FFNN = FeedForward(input_dim=3 * 2 * hidden_dim, num_layers=2, hidden_dims=[hidden_dim, 1],
                                       activations=nn.ReLU())

        # FFNN for attention to each pair
        self.pair_attention = nn.Linear(in_features=12 * hidden_dim, out_features=1)

        # Scoring pairwise sentiment: linear score approach
        '''
        self.final_sentiment_score = FeedForward(input_dim=15 * hidden_dim, num_layers=2,
                                                    hidden_dims=[hidden_dim, num_labels],
                                                    activations=nn.ReLU())
        '''
        self.final_sentiment_score = nn.Linear(in_features=17 * hidden_dim, out_features=num_labels)
        #  self.pairwise_sentiment_score = nn.Linear(in_features=12 * hidden_dim, out_features=num_labels)
        # '''

    def forward(self, word_vec, feature_vec, holder_target_vec, lengths=None,
                holder_inds=None, target_inds=None, holder_lengths=None, target_lengths=None,
                co_occur_feature=None, holder_rank=None, target_rank=None):
        # Apply embeddings & prepare input
        word_embeds_vec = self.word_embeds(word_vec)
        feature_embeds_vec = self.polarity_embeds(feature_vec)
        # print(str(word_embeds_vec.size()) + " " + str(feature_embeds_vec.size()))

        # [word embeddings, feature embeddings]
        lstm_input = torch.cat((word_embeds_vec, feature_embeds_vec), 2)
        # print(lstm_input.size())
        #        total_length = lstm_input.size(1)  # get the max sequence length

        # Mask out padding
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            # print(lengths)
            lstm_input = pack_padded_sequence(lstm_input, lengths, batch_first=True)
            # print(lstm_input.data.size())

        # Pass through lstm, encoding words
        lstm_out, _ = self.lstm(lstm_input)

        # Re-apply padding
        if lengths is not None:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]

        # Dimension: batch_size, sequence_len, 2 * hidden_dim (since concatenated forward w/ backward)
        lstm_out = self.dropout(lstm_out)

        # Mask encoded words to isolate holders
        holder_mask = (holder_inds[:, :, 0] >= 0).long()
        # holders = concatenated [start token, end token] across hidden dimension
        # Dimension: batch_size, # of holder mentions, 2 * (2 * hidden_dim)--both endpoints
        endpoint_holder = self._endpoint_span_extractor(lstm_out, holder_inds, span_indices_mask=holder_mask)
        # Dimension: batch_size, # of holder mentions, 2 * hidden_dim
        attended_holder = self._attentive_span_extractor(lstm_out, holder_inds, span_indices_mask=holder_mask)
        # Shape: (batch_size, # of holder mentions, 3 * (2 * hidden_dim))
        holders = torch.cat([endpoint_holder, attended_holder], -1)

        # Mask encoded words to isolate targets
        target_mask = (target_inds[:, :, 0] >= 0).long()
        # targets = concatenated [start token, end token] across hidden dimension
        # Dimension: batch_size, # of target mentions, 2 * (2 * hidden_dim)
        endpoint_target = self._endpoint_span_extractor(lstm_out, target_inds, span_indices_mask=target_mask)
        # Dimension: batch_size, # of target mentions, 2 * hidden_dim
        attended_target = self._attentive_span_extractor(lstm_out, target_inds, span_indices_mask=target_mask)
        # Shape: (batch_size, # of target mentions, 3 * (2 * hidden_dim))
        targets = torch.cat([endpoint_target, attended_target], -1)

        # Shape: (b, h * t, 6 * d)
        holders_repeat = holders.repeat(1, targets.size()[1], 1)  # Repeat along dimension 2
        # Shape: (b, h, 6 * d * t)
        targets_repeat = targets.repeat(1, 1, holders.size()[1])  # Repeat along dimension 2
        # Shape: (b, h * t, 6 * d)
        targets_repeat = targets_repeat.view(targets.size()[0], -1, targets.size()[2])  # View along dimension 1
        # all_pairs =
        # [H1, T1]
        # [H2, T1]
        # [H1, T2]
        # [H1, T2]
        # Shape: (b, h * t, 12 * d)
        all_pairs = torch.cat([holders_repeat, targets_repeat], dim=2)
        # Shape: (b, h * t, 1)
        pair_weights = self.pair_attention(all_pairs)
        # Shape: (b, h * t, 12 * d)
        pairwise_scores = torch.mul(pair_weights, all_pairs)
        # Shape: (b, 12 * d)
        pairwise_scores = pairwise_scores.sum(1)
        
        # Apply embeds for co-occur feature
        # Shape: (batch_size, hidden_dim)
        co_occur_feature[co_occur_feature >= 10] = 9
        co_occur_embeds_vec = self.co_occur_embeds(co_occur_feature)

        # Holder & target lengths features
        #   Holder lengths
        holder_lengths[holder_lengths >= num_mentions_cats] = num_mentions_cats
        holder_lengths = torch.add(holder_lengths, -1)
        num_holder_embeds_vec = self.num_holder_mention_embeds(holder_lengths)
        #   Target lengths
        target_lengths[target_lengths >= num_mentions_cats] = num_mentions_cats
        target_lengths = torch.add(target_lengths, -1)
        num_target_embeds_vec = self.num_holder_mention_embeds(target_lengths)
        #   min(Holder, Target) lengths
        min_lengths = torch.min(holder_lengths, target_lengths)  # already subtracted 1 from holder & target lengths
        min_lengths[min_lengths >= num_mentions_cats] = num_mentions_cats
        min_embeds_vec = self.min_mention_embeds(min_lengths)

        # Holder & target ranks
        holder_rank[holder_rank >= 5] = 5  # all ranks >= 5 get mapped to 6
        holder_rank = torch.add(holder_rank, -1)  # start indices at 0
        target_rank[target_rank >= 5] = 5
        target_rank = torch.add(target_rank, -1)  # start indices at 0
        holder_rank_vec = self.holder_rank_embeds(holder_rank)
        target_rank_vec = self.target_rank_embeds(target_rank)

        # Shape: (batch_size, 3 * hidden_dim + 12 * hidden_dim)
        final_rep = torch.cat([pairwise_scores, co_occur_embeds_vec, num_holder_embeds_vec, num_target_embeds_vec, holder_rank_vec, target_rank_vec], dim=-1)
        final_rep = self.dropout(final_rep)  # dropout

        output = self.final_sentiment_score(final_rep)
        log_probs = F.log_softmax(output, dim=1)  # Shape: b x 3

        return log_probs  # , alphas

