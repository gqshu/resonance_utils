"""Seamless-loop assembly: equal-power crossfades, chaining clips, closing a loop to an exact length."""

import numpy as np


def _fades(xf):
    w = np.linspace(0, np.pi / 2, xf)[:, None]
    return np.sin(w), np.cos(w)


def chain(clips, order, xf):
    """Concatenate clips in order with equal-power crossfades of xf samples."""
    fin, fout = _fades(xf)
    out = clips[order[0]].copy()
    for k in order[1:]:
        c = clips[k]
        out[-xf:] = out[-xf:] * fout + c[:xf] * fin
        out = np.concatenate([out, c[xf:]], 0)
    return out


def loopable_samples(clips, xf):
    """Longest seamless loop the clips can make (one extra crossfade closes the loop)."""
    return sum(len(c) for c in clips) - len(clips) * xf


def close_loop(x, n_out, xf):
    """Cut x to exactly n_out samples, blending x[n_out:n_out+xf] into the head -> seamless."""
    if n_out + xf > len(x):
        raise ValueError("not enough material for the requested loop length")
    fin, fout = _fades(xf)
    body = x[:n_out].copy()
    body[:xf] = x[:xf] * fin + x[n_out:n_out + xf] * fout
    return body
