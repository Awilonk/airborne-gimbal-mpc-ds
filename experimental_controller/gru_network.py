"""Frozen joint 40-input / 2-output GRU; NumPy-only Python 3.8 runtime.

Artifacts use PyTorch's r, z, n gate order and separate input/recurrent
biases. In particular, the reset gate multiplies the recurrent candidate
bias as well as its matrix product (PyTorch's reset-after convention).
"""
import hashlib
import io

import numpy as np

from paper_features import PaperFeatures


def _finite_real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('{} must contain finite real numbers'.format(name))
    result = np.array(array, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('{} cannot be represented as finite float64'.format(name))
    return result


class GRUNetwork:
    def __init__(self, path):
        # Hash the same bytes that are parsed, including provenance metadata.
        with open(path, 'rb') as stream:
            payload = stream.read()
        self.weight_sha256 = hashlib.sha256(payload).hexdigest()
        names = ('Wih', 'Whh', 'bih', 'bhh', 'Wout', 'bout',
                 'x_mu', 'x_sd', 'y_mu', 'y_sd')
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            required = set(names) | {'kind', 'training_validated'}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError('missing GRU artifact fields: {}'.format(', '.join(sorted(missing))))
            kind = np.asarray(archive['kind'])
            accepted = np.asarray(archive['training_validated'])
            if kind.shape != () or kind.item() != 'paper40_gru_residual':
                raise ValueError('artifact kind must be paper40_gru_residual')
            if accepted.shape != () or accepted.dtype.kind != 'b' or not accepted.item():
                raise ValueError('GRU artifact requires training_validated=True')
            arrays = {key: _finite_real(archive[key], key) for key in names}
            self.meta = {'kind': kind.item(), 'training_validated': True}
            for key in ('eps_max', 'eps_rate', 'D_max', 'hidden_size'):
                if key in archive.files:
                    self.meta[key] = _finite_real(archive[key], key).tolist()
            if 'source' in archive.files:
                source = np.asarray(archive['source'])
                if source.dtype.kind not in 'US':
                    raise ValueError('source must contain text')
                self.meta['source'] = source.astype(str).tolist()

        Whh = arrays['Whh']
        if Whh.ndim != 2 or Whh.shape[1] <= 0:
            raise ValueError('invalid GRU recurrent dimensions')
        self.H = int(Whh.shape[1])
        shapes = {'Wih': (3 * self.H, 40), 'Whh': (3 * self.H, self.H),
                  'bih': (3 * self.H,), 'bhh': (3 * self.H,),
                  'Wout': (2, self.H), 'bout': (2,),
                  'x_mu': (40,), 'x_sd': (40,), 'y_mu': (2,), 'y_sd': (2,)}
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError('{} must have shape {}'.format(key, shape))
        if 'hidden_size' in self.meta and self.meta['hidden_size'] != self.H:
            raise ValueError('hidden_size disagrees with GRU weight dimensions')
        if np.any(arrays['x_sd'] <= 0) or np.any(arrays['y_sd'] <= 0):
            raise ValueError('GRU normalization scales must be positive')
        norm = np.linalg.norm(arrays['Wout'], 2)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('invalid output spectral norm')
        self.Wih, self.Whh = arrays['Wih'], Whh
        self.bih, self.bhh = arrays['bih'], arrays['bhh']
        self.Wbar, self.bout = arrays['Wout'] / norm, arrays['bout']
        self.mu, self.sd = arrays['x_mu'], arrays['x_sd']
        self.y_mu, self.y_sd = arrays['y_mu'], arrays['y_sd']
        with np.errstate(over='ignore', invalid='ignore'):
            self.raw_bound_l2 = float(
                np.linalg.norm(self.y_sd[:, None] * self.Wbar, 2) * np.sqrt(self.H)
                + np.linalg.norm(self.y_sd * self.bout + self.y_mu))
        if not np.isfinite(self.raw_bound_l2):
            raise ValueError('nonfinite GRU output bound')
        self.reset()

    def reset(self):
        self.h = np.zeros(self.H)
        self.features = PaperFeatures()
        self.last_psi = np.zeros(40)

    def step(self, q_deg, w_deg_s, u_prev):
        # Validate before the feature buffer can be changed.
        inputs = [_finite_real(value, name) for value, name in
                  ((q_deg, 'q_deg'), (w_deg_s, 'w_deg_s'), (u_prev, 'u_prev'))]
        if any(value.shape != (2,) for value in inputs):
            raise ValueError('two finite axes required')
        return self.step_psi(self.features.step(*inputs))

    def step_psi(self, psi):
        x = _finite_real(psi, 'psi')
        if x.shape != (40,):
            raise ValueError('finite 40-vector required')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            input_gates = self.Wih @ ((x - self.mu) / self.sd) + self.bih
            hidden_gates = self.Whh @ self.h + self.bhh
            H = self.H

            def sigmoid(value):
                exponential = np.exp(-np.abs(value))
                return np.where(value >= 0, 1. / (1. + exponential),
                                exponential / (1. + exponential))

            r = sigmoid(input_gates[:H] + hidden_gates[:H])
            z = sigmoid(input_gates[H:2 * H] + hidden_gates[H:2 * H])
            n = np.tanh(input_gates[2 * H:] + r * hidden_gates[2 * H:])
            hidden = (1. - z) * n + z * self.h
            result = (self.Wbar @ hidden + self.bout) * self.y_sd + self.y_mu
        if not np.isfinite(hidden).all() or not np.isfinite(result).all():
            raise FloatingPointError('nonfinite GRU network output')
        self.h = hidden
        self.last_psi = x
        return result

    def diagnostics(self):
        return {'architecture': 'GRU', 'hidden_state': self.h.tolist(),
                'hidden_size': self.H, 'weight_sha256': self.weight_sha256,
                'kind': self.meta['kind'], 'training_validated': True,
                'raw_bound_l2': self.raw_bound_l2}
