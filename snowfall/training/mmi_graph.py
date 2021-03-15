# Copyright (c)  2020  Xiaomi Corp.       (author: Fangjun Kuang)

import k2
import torch
from typing import Iterable
from typing import List
from typing import Tuple

from snowfall.common import get_phone_symbols
from .ctc_graph import build_ctc_topo


def create_bigram_phone_lm(phones: List[int]) -> k2.Fsa:
    '''Create a bigram phone LM.
    The resulting FSA (P) has a start-state and a state for
    each phone 1, 2, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.

    Caution:
      blank is not a phone.

    Args:
      A list of phone IDs.

    Returns:
      An FSA representing the bigram phone LM.
    '''
    assert 0 not in phones
    final_state = len(phones) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {phones[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {phones[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
    return k2.Fsa.from_str(rules)


class MmiTrainingGraphCompiler(object):

    def __init__(self,
                 L_inv: k2.Fsa,
                 phones: k2.SymbolTable,
                 words: k2.SymbolTable,
                 device: torch.device,
                 topo_builder_fn=build_ctc_topo,
                 oov: str = '<UNK>'):
        '''
        Args:
          L_inv:
            Its labels are words, while its aux_labels are phones.
        phones:
          The phone symbol table.
        words:
          The word symbol table.
        oov:
          Out of vocabulary word.
        '''

        L_inv = L_inv.to(device)

        if L_inv.properties & k2.fsa_properties.ARC_SORTED != 0:
            L_inv = k2.arc_sort(L_inv)

        assert L_inv.requires_grad is False

        assert oov in words

        self.L_inv = L_inv
        self.phones = phones
        self.words = words
        self.oov_id = self.words[oov]
        self.device = device

        phone_symbols = get_phone_symbols(phones)
        phone_symbols_with_blank = [0] + phone_symbols

        H = topo_builder_fn(phone_symbols_with_blank).to(device)
        assert H.requires_grad is False
        self.H_inv = k2.arc_sort(H.invert_())

    def compile(self, texts: Iterable[str],
                P: k2.Fsa) -> Tuple[k2.Fsa, k2.Fsa]:
        '''Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces.
          P:
            The bigram phone LM created by :func:`create_bigram_phone_lm`.
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec with the same
              shape of the `num_graph`.
        '''
        assert P.device == self.device
        P_with_self_loops = k2.add_epsilon_self_loops(P)

        HP = k2.intersect(
            self.H_inv,
            P_with_self_loops,
            treat_epsilons_specially=False
        ).invert()
        HP = k2.arc_sort(HP)

        num_graphs = self.build_num_graphs(texts)
        num_graphs_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            num_graphs)
        num_graphs_with_self_loops = k2.arc_sort(num_graphs_with_self_loops)

        num = k2.compose(HP,
                         num_graphs_with_self_loops,
                         treat_epsilons_specially=False)
        num = k2.arc_sort(num)

        HP_vec = k2.create_fsa_vec([HP.detach()])
        indexes = torch.zeros(len(texts),
                              dtype=torch.int32,
                              device=self.device)
        den = k2.index_fsa(HP_vec, indexes)

        return num, den

    def build_num_graphs(self, texts: List[str]) -> k2.Fsa:
        '''Convert transcript to an Fsa with the help of lexicon
        and word symbol table.

        Args:
          texts:
            Each element is a transcript containing words separated by spaces.
            For instance, it may be 'HELLO SNOWFALL', which contains
            two words.

        Returns:
          Return an FST (FsaVec) corresponding to the transcript. Its `labels` are
          phone IDs and `aux_labels` are word IDs.
        '''
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split(' '):
                if word in self.words:
                    word_ids.append(self.words[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        fsa = k2.add_epsilon_self_loops(fsa)
        assert fsa.device == self.device
        num_graphs = k2.intersect(self.L_inv,
                                  fsa,
                                  treat_epsilons_specially=False).invert_()
        num_graphs = k2.arc_sort(num_graphs)
        return num_graphs

