import os
import numpy as np
from PIL import Image
import math
import time


def print_current_loss(start_time, niter_state, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('epoch: %3d niter: %6d sub_epoch: %2d inner_iter: %4d' % (epoch, niter_state, sub_epoch, inner_iter), end=" ")

    # message = '%s niter: %d completed: %3d%%)' % (time_since(start_time, niter_state / total_niters),
    #                                             niter_state, niter_state / total_niters * 100)
    now = time.time()
    message = '%s'%(as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    message += ' sl_length:%2d tf_ratio:%.2f'%(sl_steps, tf_ratio)
    print(message)

def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None, tf_ratio=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)

    # message += ' tf_ratio:%.2f'%(tf_ratio)

    print(message)