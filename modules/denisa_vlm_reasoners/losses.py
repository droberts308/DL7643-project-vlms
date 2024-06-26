# Edited from https://github.com/merlresearch/SMART
import torch.nn as nn

import text_encoder as gv


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, a, b, pids):
        loss = 0
        for key in a.keys():
            idx = pids == int(key)
            if int(key) not in gv.SEQ_PUZZLES:
                loss += self.criterion(
                    a[key], b[idx, 0]
                )  # risky if idx and key entries are not matched. but then we will encouter an exception.
            else:
                seq_loss = 0
                for i in range(len(a[key])):
                    seq_loss += self.criterion(a[key][i], b[idx, i])
                seq_loss /= len(a[key])
                loss += seq_loss
        loss = loss / len(a.keys())
        return loss

    def forward(self, a, b, pids=None):
        loss = self.compute_loss(a, b.long(), pids)
        return loss
