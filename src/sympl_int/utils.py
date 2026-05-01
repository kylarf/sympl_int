from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

type ConservativeForce = Callable[[NDArray[np.float64]], NDArray[np.float64]]
type FloatNDArray = NDArray[np.float64]


def promote_to_3d(vec: FloatNDArray) -> FloatNDArray:
    r = vec.copy()
    if r.ndim == 1:
        if r.shape[0] == 2:
            r = np.array([r[0], r[1], 0.0], dtype=r.dtype)
        elif r.shape[0] != 3:
            raise ValueError("r must have shape (2,), (3,), (N,2), or (N,3)")
    elif r.ndim == 2:
        if r.shape[-1] == 2:
            r = np.column_stack((r, np.zeros(r.shape[0], dtype=r.dtype)))
        elif r.shape[-1] != 3:
            raise ValueError("r must have shape (2,), (3,), (N,2), or (N,3)")
    else:
        raise ValueError("r must have ndim 1 or 2")
    return r


@dataclass
class OrbitalElements:
    mu: np.floating
    semimajor_axis: FloatNDArray
    eccentricity: FloatNDArray
    inclination: FloatNDArray
    lon_ascending_node: FloatNDArray
    arg_periapsis: FloatNDArray
    true_anomaly: FloatNDArray
    angular_momentum: FloatNDArray

    @classmethod
    def from_rv(cls, r: FloatNDArray, v: FloatNDArray, mu: np.floating):
        r = promote_to_3d(r)
        v = promote_to_3d(v)

        norm_r = np.linalg.norm(r, axis=-1)
        norm_v = np.linalg.norm(v, axis=-1)

        h = np.cross(r, v)
        norm_h = np.linalg.norm(h, axis=-1)
        n = np.cross(np.array([0.0, 0.0, 1.0]), h)

        e_vec = (1 / mu) * (
            (norm_v**2 - mu / norm_r)[..., None] * r - np.vecdot(r, v)[..., None] * v
        )
        norm_e = np.linalg.norm(e_vec, axis=-1)

        inclination = np.arccos(h[..., 2] / norm_h)
        lon_ascending_node = np.atan2(n[..., 1], n[..., 0])
        arg_periapsis = np.atan2(
            np.vecdot(np.cross(n, e_vec), h) / norm_h,
            np.vecdot(n, e_vec),
        )
        true_anomaly = np.atan2(
            np.vecdot(np.cross(e_vec, r), h) / norm_h,
            np.vecdot(e_vec, r),
        ) % (2 * np.pi)
        semimajor_axis = 1 / (2 / norm_r - norm_v**2 / mu)

        return cls(
            mu,
            semimajor_axis,
            norm_e,
            inclination,
            lon_ascending_node,
            arg_periapsis,
            true_anomaly,
            h,
        )

    def to_rv(self) -> tuple[FloatNDArray, FloatNDArray]:
        semilatus_rectum = self.semimajor_axis * (1 - self.eccentricity**2)

        r = semilatus_rectum / (1 + self.eccentricity * np.cos(self.true_anomaly))
        r_perifocal = r[..., None] * np.stack(
            (
                np.cos(self.true_anomaly),
                np.sin(self.true_anomaly),
                np.zeros_like(self.true_anomaly),
            ),
            axis=-1
        )
        v_perifocal = np.sqrt(self.mu / semilatus_rectum)[..., None] * np.stack(
            (
                -np.sin(self.true_anomaly),
                self.eccentricity + np.cos(self.true_anomaly),
                np.zeros_like(self.true_anomaly),
            ),
            axis=-1
        )

        rotation = Rotation.from_euler(
            "ZXZ",
            np.stack(
                (self.lon_ascending_node, self.inclination, self.arg_periapsis),
                axis=-1,
            )
        )

        return rotation.apply(r_perifocal), rotation.apply(v_perifocal)

    @property
    def period(self):
        return 2 * np.pi * np.sqrt(self.semimajor_axis**3 / self.mu)

    @property
    def energy(self):
        return -self.mu / 2 / self.semimajor_axis

    @property
    def eccentric_anomaly(self):
        return np.atan2(
            np.sqrt(1 - self.eccentricity**2) * np.sin(self.true_anomaly),
            self.eccentricity + np.cos(self.true_anomaly),
        )

    @property
    def mean_motion(self):
        return 2 * np.pi / self.period

    @property
    def mean_anomaly(self):
        return self.eccentric_anomaly - self.eccentricity * np.sin(
            self.eccentric_anomaly
        )
