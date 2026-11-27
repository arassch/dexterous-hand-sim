"""Volumetric tet meshes for fingertip pads.

A cube grid is mapped to a rounded box (blend of the cube and its sphere mapping), split
into 6 Kuhn tets per hex, optionally curled at one end so the pad wraps over the fingertip.
"""
import itertools

import numpy as np

_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def _cube_to_sphere(x, y, z):
    return np.array([x * np.sqrt(1 - y * y / 2 - z * z / 2 + y * y * z * z / 3),
                     y * np.sqrt(1 - z * z / 2 - x * x / 2 + z * z * x * x / 3),
                     z * np.sqrt(1 - x * x / 2 - y * y / 2 + x * x * y * y / 3)])


def pad_tets(R, n=5, boxiness=0.0, curl_axis=1, curl_dir=-1, curl_start=None, curl_radius=None,
             curl_towards=0):
    """Rounded-box pad with half-extents R.

    boxiness: 0 = ellipsoid, 1 = box.
    Curl: vertices beyond `curl_start` along `curl_dir * axis[curl_axis]` are bent towards
    +axis[curl_towards] on a circle of `curl_radius`, so the pad wraps over the fingertip end.
    Returns (points [N,3], tets [M,4], grid coords [N,3] in [-1,1]) with positively oriented tets.
    """
    R = np.asarray(R, float)
    g = np.linspace(-1, 1, n)
    idx, P, G = {}, [], []
    for i, j, k in itertools.product(range(n), repeat=3):
        c = np.array([g[i], g[j], g[k]])
        p = (boxiness * c + (1 - boxiness) * _cube_to_sphere(*c)) * R
        if curl_start is not None:
            s = curl_dir * p[curl_axis] - curl_start
            if s > 0:   # bend: arc length s on a circle of radius (curl_radius - depth)
                depth = -p[curl_towards]                       # distance from the pad's inner plane
                rr = curl_radius + depth
                th = s / curl_radius
                p[curl_axis] = curl_dir * (curl_start + rr * np.sin(th))
                p[curl_towards] = -(rr * np.cos(th) - curl_radius)
        idx[i, j, k] = len(P)
        P.append(p)
        G.append(c)
    P = np.array(P)
    T = []
    for i, j, k in itertools.product(range(n - 1), repeat=3):
        c = [idx[i + a, j + b, k + cc] for a, b, cc in
             [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]]
        for t in _KUHN:
            v = [c[q] for q in t]
            if np.linalg.det(np.array([P[v[1]] - P[v[0]], P[v[2]] - P[v[0]], P[v[3]] - P[v[0]]])) < 0:
                v[2], v[3] = v[3], v[2]
            T.append(v)
    return P, np.array(T), np.array(G)
