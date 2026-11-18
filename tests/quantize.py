"""Verify signed nibble serialization and covariance-aware error minimization."""

import numpy as np
from training.export import quantize_grouped, refine_block


def main():
    rng = np.random.default_rng(83742)
    group = rng.standard_normal((128, 32)).astype("float32")
    group[0] = 0
    group[1, 1] = 6
    group[2, 2] = -7
    samples = rng.standard_normal((128, 54, 32)).astype("float32")
    cov = samples.transpose(0, 2, 1) @ samples / 54 + 0.25 * np.eye(32, dtype="float32")
    scale, q = quantize_grouped(
        group, 4, True, np.diagonal(cov, axis1=1, axis2=2), True
    )

    def errors(s, codes):
        d = group - s * codes
        return np.einsum("bi,bij,bj->b", d, cov, d)

    old = errors(scale, q)
    step, codes = refine_block(group, scale.copy(), q, cov, -8)
    assert np.all(errors(step, codes) <= old + 1e-5)
    assert (step < 0).any() and (step > 0).any()
    assert codes.min() == -8 and codes.max() == 7 and np.isfinite(step).all()
    biased = (codes + 8).astype("uint8")
    packed = biased[:, :16] | (biased[:, 16:] << 4)
    restored = np.concatenate((packed & 15, packed >> 4), 1).astype("int8") - 8
    np.testing.assert_array_equal(codes, restored)
    print(
        "All 16 codes round-trip. Every refined group preserved or reduced calibration error."
    )


if __name__ == "__main__":
    main()
