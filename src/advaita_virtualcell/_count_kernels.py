"""Numerical kernels retained from the validated count renderer."""

import numpy as np
from scipy import sparse


def _lr_round(x):
    """Round a nonnegative vector by largest remainder."""
    f = np.floor(x)
    tot = int(round(float(x.sum())))
    short = tot - int(f.sum())
    if short > 0:
        idx = np.argsort(-(x - f), kind="stable")[:short]
        f[idx] += 1
    return f.astype(np.int64)


def _stoch_round(x, rng, keep_pos):
    f = np.floor(x)
    y = f + (rng.random(x.shape) < x - f)
    return np.where(keep_pos & (y < 1), 1, y).astype(np.int64)


def _mean_round(Q, L, X, T, rng):
    """Round paired expression values and repair their arithmetic means."""
    n, G = Q.shape
    Z = Q * L[:, None]
    Y = np.floor(Z).astype(np.int64)
    rem = Z - Y
    tot = np.rint(Z.sum(0)).astype(np.int64)
    short = tot - Y.sum(0)
    order = np.argsort(-rem - 1e-09 * rng.random(rem.shape), axis=0)
    for g in np.flatnonzero(short > 0):
        Y[order[: short[g], g], g] += 1
    invL = 1.0 / L
    big = np.argsort(-L)
    small = np.argsort(L)
    m0 = (X / L[:, None]).sum(0)
    m1 = (Y * invL[:, None]).sum(0)
    step = invL[small[0]] - invL[big[0]]
    for g in np.flatnonzero(np.abs(m1 - m0) > step):
        d = m1[g] - m0[g]
        for _ in range(8):
            if d < -step / 2:
                src = next((c for c in big if Y[c, g] > 0), None)
                if src is None:
                    break
                dst = small[0] if small[0] != src else small[1]
                Y[src, g] -= 1
                Y[dst, g] += 1
                d += invL[dst] - invL[src]
            elif d > step / 2:
                src = next((c for c in small if Y[c, g] > 0), None)
                if src is None:
                    break
                dst = big[0] if big[0] != src else big[1]
                Y[src, g] -= 1
                Y[dst, g] += 1
                d += invL[dst] - invL[src]
            else:
                break
    return Y


def pair_to_totals(X, L, T, rng, iters=22, mode="stoch"):
    """Pair normalized expression ranks with cell libraries to approach target totals."""
    n, G = X.shape
    q = X / L[:, None]
    cur = X.sum(0).astype(np.float64)
    rq = np.argsort(np.argsort(q + 1e-12 * rng.random(q.shape), axis=0), axis=0) / max(n - 1, 1)
    rL = (np.argsort(np.argsort(L)) / max(n - 1, 1))[:, None]
    qs = np.sort(q, axis=0)
    sgn = np.sign(T - cur)

    def realise(lam):
        key = lam[None, :] * np.where(sgn[None, :] >= 0, rL, 1 - rL) + (1 - lam[None, :]) * rq
        order = np.argsort(key, axis=0)
        Q = np.empty_like(q)
        np.put_along_axis(Q, order, qs, axis=0)
        return Q

    lo = np.zeros(G)
    hi = np.ones(G)
    Qhi = realise(hi)
    tot_hi = (Qhi * L[:, None]).sum(0)
    reach = np.abs(tot_hi - cur) >= np.abs(T - cur)
    for _ in range(iters):
        mid = (lo + hi) / 2
        tot = (realise(mid) * L[:, None]).sum(0)
        past = np.abs(tot - cur) >= np.abs(T - cur)
        hi = np.where(past, mid, hi)
        lo = np.where(past, lo, mid)
    lam = np.where(reach, hi, 1.0)
    Q = realise(lam)
    Y = _mean_round(Q, L, X, T, rng) if mode == "mean" else _stoch_round(Q * L[:, None], rng, Q > 0)
    return (Y, reach)


def fix_totals(Y, T, L, rng, allow_scale):
    """Correct residual column errors with integer additions and removals."""
    for g in np.flatnonzero(Y.sum(0) != T):
        d = int(T[g] - Y[:, g].sum())
        if d > 0:
            Y[:, g] += rng.multinomial(d, L / L.sum())
        else:
            col = Y[:, g]
            k = min(-d, int(col.sum()))
            if k > 0:
                Y[:, g] = col - rng.multivariate_hypergeometric(col, k)
    return Y


def shift_gene(col, f, rng):
    if f < 1:
        return rng.binomial(col, f)
    p0 = max((col == 0).mean(), 1e-06)
    conv = (col == 0) & (rng.random(col.shape) < 1 - p0 ** (f - 1))
    up = _stoch_round(col * f, rng, col > 0)
    return np.where(conv, 1, up)


def tail_shift_gene(col, f, tau, rng):
    """Apply a mean shift to the highest-count fraction of cells."""
    tot = col.sum()
    if tot <= 0 or f == 1.0:
        return col.copy()
    k = max(1, int(round(tau * len(col))))
    idx = np.argsort(col, kind="stable")[-k:]
    out = col.copy()
    D = (f - 1.0) * tot
    if rng is None:
        if D > 0:
            w = col[idx] + 1.0
            out[idx] += _lr_round(int(round(D)) * w / w.sum())
        else:
            top = col[idx]
            p = min(0.95, -D / max(top.sum(), 1))
            out[idx] = top - _lr_round(top * p)
        return out
    if D > 0:
        w = col[idx] + 1.0
        out[idx] += rng.multinomial(int(round(D)), w / w.sum())
    else:
        top = col[idx]
        p = min(0.95, -D / max(top.sum(), 1))
        out[idx] = top - rng.binomial(top, p)
    return out


def desired_mean(base, call, own, own_keep=0.3, mref=None):
    """Construct normalized arithmetic-mean targets from signed expression effects."""
    X = sparse.csr_matrix(base)
    L = np.asarray(X.sum(1)).ravel()
    m = np.asarray((sparse.diags(1 / L) @ X).mean(0)).ravel()
    if mref is not None:
        dg = (
            np.asarray(call[0])
            if len(call) <= 3
            else np.concatenate([np.asarray(call[0]), np.asarray(call[3])]).astype(int)
        )
        m = m.copy()
        m[dg] = mref[dg]
    fac = np.ones(X.shape[1])
    fac[call[0]] = np.exp2(np.asarray(call[1]) * np.asarray(call[2]))
    if len(call) > 3:
        fac[call[3]] = np.exp2(np.asarray(call[4]) * np.asarray(call[5]))
    if own >= 0:
        fac[own] *= own_keep
    raw = m * fac
    return raw / raw.sum()


def mean_error(Y, target, mask):
    L = Y.sum(1).astype(float)
    if (L <= 0).any():
        return float("inf")
    m = (Y / L[:, None]).mean(0)
    return float(np.mean(abs(np.log2((m[mask] + 1e-09) / (target[mask] + 1e-09)))))


def readout_repair(
    y,
    base,
    call,
    own,
    *,
    passes=2,
    max_moves=16,
    fraction=0.25,
    support=False,
    distributed=False,
    own_keep=0.3,
    dense=False,
    moderate=False,
    mref=None,
):
    """Improve normalized design means while preserving exact gene totals."""
    moderate_only = moderate
    Y = sparse.csr_matrix(y).toarray().astype(np.int64)
    initial_Y = Y.copy()
    total = Y.sum(0)
    target = desired_mean(base, call, own, own_keep, mref)
    mask = target >= 5e-06
    if own >= 0:
        mask[own] = False
    genes = np.asarray(call[0])
    if dense and len(call) > 3:
        genes = np.unique(np.concatenate([genes, np.asarray(call[3])]))
    genes = genes[mask[genes]]
    error = mean_error(Y, target, mask)
    initial = error
    accepted = 0
    moved = 0
    initial_L = Y.sum(1)
    touch = np.zeros((len(Y), len(genes)), np.uint8)
    rank = np.argsort(np.argsort(initial_L, kind="stable"), kind="stable")
    moderate = (rank >= int(0.1 * len(Y))) & (rank < int(0.9 * len(Y)))
    for _ in range(passes):
        Z = Y.copy()
        proposed_touch = touch.copy()
        L = Z.sum(1).astype(float)
        inv = 1 / L
        wanted = target[genes] * len(Z)
        cur = (Z[:, genes] * inv[:, None]).sum(0)
        goal = cur + fraction * (wanted - cur)
        for __ in range(max_moves):
            cols = Z[:, genes]
            delta = goal - (cols * inv[:, None]).sum(0)
            up = delta > 0
            donor = (cols > (1 if support else 0)) & (L[:, None] > 1)
            recv = cols > 0 if support else np.ones(cols.shape, bool)
            if distributed:
                allow = (proposed_touch < 2) & moderate[:, None]
                donor &= allow
                recv &= allow
            elif moderate_only:
                donor &= moderate[:, None]
                recv &= moderate[:, None]
            src_up = np.argmin(np.where(donor, inv[:, None], np.inf), axis=0)
            src_dn = np.argmax(np.where(donor, inv[:, None], -np.inf), axis=0)
            dst_up = np.argmax(np.where(recv, inv[:, None], -np.inf), axis=0)
            dst_dn = np.argmin(np.where(recv, inv[:, None], np.inf), axis=0)
            src = np.where(up, src_up, src_dn)
            dst = np.where(up, dst_up, dst_dn)
            colidx = np.arange(len(genes))
            step = inv[dst] - inv[src]
            ok = (
                donor[src, colidx]
                & recv[dst, colidx]
                & (src != dst)
                & (np.sign(step) == np.sign(delta))
                & (abs(delta - step) < abs(delta))
            )
            jj = genes[ok]
            ss = src[ok]
            dd = dst[ok]
            kk = colidx[ok]
            if not len(jj):
                break
            Z[ss, jj] -= 1
            Z[dd, jj] += 1
            proposed_touch[ss, kk] += 1
            proposed_touch[dd, kk] += 1
            L = Z.sum(1).astype(float)
            if (L <= 0).any():
                break
            inv = 1 / L
        e = mean_error(Z, target, mask)
        if e < error and (Z >= 0).all() and (Z.sum(1) > 0).all():
            moved += int(abs(Z - Y).sum() // 2)
            Y = Z
            error = e
            accepted += 1
            touch = proposed_touch
    np.testing.assert_array_equal(Y.sum(0), total)
    changed_support = int(np.count_nonzero((Y > 0) != (initial_Y > 0)))
    if support:
        assert changed_support == 0
    if distributed:
        assert int(abs(Y[:, genes] - initial_Y[:, genes]).max(initial=0)) <= 2
    out = sparse.csr_matrix(Y.astype(np.int32))
    out.eliminate_zeros()
    out.sort_indices()
    return (
        out,
        dict(
            repair_error_before=initial,
            repair_error_after=error,
            repair_passes=accepted,
            repair_molecules=moved,
            support_changes=changed_support,
            max_cell_library_relative_change=float(np.max(abs(Y.sum(1) - initial_L) / initial_L)),
            max_entry_change=int(abs(Y - initial_Y).max(initial=0)),
        ),
    )


def shape_shift_gene(col, L, s, c, eps, rng, mref=None):
    """Adjust normalized expression dispersion and rescale to a target mean."""
    q = col / L
    m = q.mean()
    if m <= 0:
        return col.copy()
    lam = max(0.0, 1.0 - c) if s > 0 else 1.0 + c
    q2 = np.maximum(m + lam * (q - m), 0.0)
    q2 *= (m if mref is None else mref) * 2.0 ** (s * eps) / max(q2.mean(), 1e-30)
    x = q2 * L
    if rng is None:
        return _lr_round(x)
    f = np.floor(x)
    return (f + (rng.random(len(x)) < x - f)).astype(np.int64)
