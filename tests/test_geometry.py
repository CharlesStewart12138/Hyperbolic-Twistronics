import numpy as np

from geometry.build_orbit_and_frames import build_orbit, frame_at, parallel_transport, validation_checks


def test_coordinate_frames_and_transport() -> None:
    points, _ = build_orbit(1, 1.0)
    frames = np.stack([frame_at(point, 1.0) for point in points])
    checks = validation_checks(points, frames, 1.0)
    assert max(float(value) for value in checks.values()) < 1.0e-9
    transported = parallel_transport(points[0], points[1], frames[0][:, 0], 1.0)
    assert np.isfinite(transported).all()

