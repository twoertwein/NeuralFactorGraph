from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils


class Lemmatizer(nn.Module):

    """
    Lemmatizer module: Still under construction
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        mlp_size,
        n_layers=2,
        dropOut=0.2,
        gpu=False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.src_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.tgt_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.

        self.enc_lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=dropOut,
            bidirectional=True,
        )
        self.dec_lstm = nn.LSTM(
            self.embedding_dim + self.hidden_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
        )

        self.proj1 = nn.Linear(2 * self.hidden_dim, self.mlp_size)
        self.proj2 = nn.Linear(self.mlp_size, self.vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Initialize hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=True),
            utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=True),
        )

    def forward(self, word):
        embeds = self.char_embeddings(word)
        lstm_out, self.hidden = self.lstm(embeds.view(len(word), 1, -1), self.hidden)
        c_space = self.proj2(self.proj1(lstm_out.view(len(word), -1)))


#         return c_space, lstm_out[-1].squeeze() # Last timestep's hidden state in last layer


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        model_type,
        sum_word_char,
        word_freq,
        sent_attn,
        langs,
        embedding_dim,
        hidden_dim,
        mlp_size,
        char_vocab_size,
        word_vocab_size,
        tagset_size,
        n_layers=2,
        dropOut=0.2,
        gpu=False,
    ):

        super().__init__()

        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.hidden_dim = hidden_dim
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.tagset_size = tagset_size
        self.n_layers = n_layers
        self.sum_word_char = sum_word_char
        self.sent_attn = sent_attn
        self.word_freq = word_freq
        self.langs = langs
        self.gpu = gpu

        # CharLSTM
        self.char_embeddings = nn.Embedding(self.char_vocab_size, self.embedding_dim)
        self.char_lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=dropOut,
            bidirectional=True,
        )

        # The linear layer that maps from hidden state space to
        # self.proj1 = nn.Linear(2 * self.hidden_dim, self.mlp_size)
        # self.proj2 = nn.Linear(self.mlp_size, self.char_vocab_size)
        self.char_hidden = self.init_hidden()

        if self.sum_word_char:
            self.word_embeddings = nn.Embedding(
                self.word_vocab_size, 2 * self.hidden_dim
            )

        if self.sent_attn:
            self.proj1 = nn.Linear(2 * self.hidden_dim, self.mlp_size, bias=False)
            self.proj2 = nn.Linear(self.mlp_size, 1, bias=False)

        self.lstm = nn.LSTM(
            self.hidden_dim * 2,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=dropOut,
            bidirectional=True,
        )

        # The linear layer that maps from hidden state space to tag space
        if self.model_type == "specific" or self.model_type == "joint":
            # self.hidden2tag = nn.ParameterList([nn.Parameter(torch.randn(2 * self.hidden_dim, self.tagset_size)) for i in range(len(self.langs))])
            self.hidden2tag_1 = nn.Linear(2 * self.hidden_dim, self.tagset_size)
            self.hidden2tag_2 = nn.Linear(2 * self.hidden_dim, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        if self.model_type == "joint":
            self.joint_tf1 = nn.Linear(
                2 * self.hidden_dim, self.tagset_size, bias=False
            )
            self.joint_tf2 = nn.Linear(self.tagset_size, len(self.langs), bias=False)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Initialize hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            utils.get_var(
                torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=self.gpu
            ),
            utils.get_var(
                torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=self.gpu
            ),
        )

    def forward(self, words, word_idxs=None, lang=None, test=False):
        """
        words: list of tensors with idxs for each character
        word_idxs: list of word_idxs
        """
        embeds = [self.char_embeddings(word) for word in words]

        if not self.sent_attn:
            char_embs = [
                self.char_lstm(embeds[i].view(len(word), 1, -1), self.char_hidden)[0][
                    -1
                ]
                for i, word in enumerate(words)
            ]
        else:
            word_attns = [
                self.char_lstm(embeds[i].view(len(word), 1, -1), self.char_hidden)[
                    0
                ].view(len(word), -1)
                for i, word in enumerate(words)
            ]
            attn_probs = [F.tanh(self.proj1(w_attn)) for w_attn in word_attns]
            attn_probs = [
                F.softmax(self.proj2(w_attn).view(w_attn.size(0)))
                for w_attn in attn_probs
            ]
            char_embs = [
                torch.sum(a.unsqueeze(1).repeat(1, w_attn.size(1)) * w_attn, 0)
                for a, w_attn in zip(attn_probs, word_attns)
            ]

        char_embs = torch.stack(char_embs).view(len(words), 1, -1)
        if self.sum_word_char:
            mask = torch.FloatTensor(
                [
                    1 if self.word_freq[int(w.cpu().numpy())] >= 5 else 0
                    for w in word_idxs
                ]
            ).cuda()
            word_embs = self.word_embeddings(word_idxs) * Variable(
                mask.unsqueeze(1).repeat(1, 2 * self.hidden_dim)
            )
            char_embs = char_embs.view(len(words), -1) + word_embs
            char_embs = char_embs.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(char_embs, self.hidden)

        if self.model_type == "specific" and lang:
            # tag_space = self.hidden2tag[lang](lstm_out.view(char_embs.size(0), -1))
            if self.langs.index(lang) == 0:
                tag_space = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
            else:
                tag_space = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))

        elif self.model_type == "joint" and lang:
            tf1 = F.tanh(self.joint_tf1(lstm_out.view(char_embs.size(0), -1)))
            tf2 = F.log_softmax(self.joint_tf2(tf1))
            if self.langs.index(lang) == 0:
                tag_space = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
            else:
                tag_space = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))

        elif self.model_type == "joint" and test:
            tf1 = F.tanh(self.joint_tf1(lstm_out.view(char_embs.size(0), -1)))
            tf2 = F.log_softmax(self.joint_tf2(tf1))
            pred_lang_scores, idxs = torch.max(tf2, 1)
            tag_space_1 = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
            tag_space_2 = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))
            tag_space = [
                tag_space_1[i] if idx == 0 else tag_space_2[i]
                for i, idx in enumerate(idxs.cpu().numpy().flatten())
            ]
            tag_space = torch.stack(tag_space)

        else:
            tag_space = self.hidden2tag(lstm_out.view(char_embs.size(0), -1))

        tag_scores = F.log_softmax(tag_space)
        if self.model_type == "joint" and not test:
            tag_scores = tag_scores + tf2[:, self.langs.index(lang)].repeat(
                1, self.tagset_size
            )
        elif self.model_type == "joint" and test:
            tag_scores = tag_scores + pred_lang_scores.repeat(1, self.tagset_size)

        return tag_scores


class BiLSTMTaggerBatched(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        char_vocab_size,
        word_vocab_size,
        tagset_sizes={},
        n_layers=2,
        dropOut=0.2,
        gpu=False,
    ):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.n_layers = n_layers
        self.gpu = gpu

        # CharLSTM
        self.char_embeddings = nn.Embedding(self.char_vocab_size, self.embedding_dim)
        self.char_lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=dropOut,
            bidirectional=True,
        )

        self.lstm = nn.LSTM(
            self.hidden_dim * 2,
            self.hidden_dim,
            num_layers=self.n_layers,
            dropout=dropOut,
            bidirectional=True,
            batch_first=True,
        )

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = torch.nn.ModuleDict(
            {
                key: nn.Linear(2 * self.hidden_dim, value)
                for key, value in tagset_sizes.items()
            }
        )

    def forward(self, sentences):
        """
        words: list (sentences) of lists (words) of tensors (character)
        """
        # get the embedding for each character
        embeds = [
            self.char_embeddings(word) for sentence in sentences for word in sentence
        ]

        # process each word individually
        lengths = [word.shape[0] for word in embeds]
        embeds = torch.nn.utils.rnn.pad_sequence(embeds, batch_first=True)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )
        words = self.char_lstm(embeds)[0]
        words, lengths = torch.nn.utils.rnn.pad_packed_sequence(words, batch_first=True)
        words = words[range(len(lengths)), lengths - 1, :]

        # process each sentence individually
        words = words.reshape(
            len(sentences), int(words.shape[0] / len(sentences)), words.shape[1]
        )
        words = self.lstm(words)[0]

        tag_space = {}
        for key, hidden2tag in self.hidden2tag.items():
            tag_space[key] = F.log_softmax(hidden2tag(words), dim=-1)
        return tag_space


class LinearChainCRF(torch.nn.Module):
    def __init__(self, states: int = -1):
        super().__init__()

        self.transition = nn.Parameter(torch.randn(states, states))
        self.source = nn.Parameter(torch.rand(states))
        self.sink = nn.Parameter(torch.rand(states))

    def viterbi_decode(self, scores):
        batch, seq, states = scores.shape

        transition = self.transition.expand(batch, -1, -1)

        alphas = torch.zeros((batch, states), device=scores.device, dtype=scores.dtype)
        best_children = torch.zeros((batch, seq, states), dtype=int)
        index = list(range(scores.shape[-1]))

        # forward pass
        for t in range(seq):
            possible_alphas = alphas[:, :, None] + scores[:, t, None]
            if t != 0:
                possible_alphas = possible_alphas + transition
            else:
                possible_alphas = possible_alphas + self.source.expand(batch, 1, -1)
            if t == seq - 1:
                possible_alphas = possible_alphas + self.sink.expand(batch, 1, -1)
            best_children[:, t, :] = torch.max(possible_alphas, dim=1).indices
            alphas = possible_alphas[:, best_children[:, t, :], index][:, 0, :]

        # backward from best state
        states = -torch.ones((batch, seq), dtype=int)
        states[:, -1] = torch.max(alphas, dim=1).indices
        for t in range(seq - 1, 0, -1):
            states[:, t - 1] = best_children[range(batch), t, states[:, t]]

        return alphas[range(batch), states[:, -1]], states

    def log_score(self, scores, states):
        batch, seq, _ = scores.shape

        score = self.source[states[:, 0]] + scores[range(batch), 0, states[:, 0]]
        for t in range(1, seq):
            score = (
                score
                + scores[range(batch), t, states[:, t]]
                + self.transition[None, states[:, t - 1], states[:, t]]
            )
        return score + self.sink[states[:, -1]]

    def neg_log_likelihood(self, scores, states):
        losses = self.log_partition(scores) - self.log_score(scores, states)
        assert (losses > -1e-10).all()
        return losses.mean()

    def log_partition(self, scores):
        batch, seq, states = scores.shape
        transition = self.transition.expand(batch, -1, -1)
        score = torch.zeros((batch, states), dtype=scores.dtype, device=scores.device)

        for t in range(seq):
            tmp = score[:, :, None] + scores[:, t, None]
            if t != 0:
                tmp = tmp + transition
            else:
                tmp = tmp + self.source.expand(batch, 1, -1)
            score = torch.logsumexp(tmp, 1)
        return torch.logsumexp(score + self.sink, 1)


class BiLSTMCRFTagger(BiLSTMTaggerBatched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.crfs = torch.nn.ModuleDict(
            {
                tag: LinearChainCRF(states=tag_size)
                for tag, tag_size in kwargs["tagset_sizes"].items()
            }
        )

    def forward(self, *args, **kwargs):
        scores = super().forward(*args)
        for tag, tag_scores in scores.items():
            scores[tag] = {
                "scores": tag_scores,
                "states": self.crfs[tag].viterbi_decode(tag_scores)[-1],
            }
        return scores

    def neg_log_likelihood(self, scores, states):
        loss = 0.0
        for tag, tag_dict in scores.items():
            loss = loss + self.crfs[tag].neg_log_likelihood(
                tag_dict["scores"], states[tag]
            )

        return loss
