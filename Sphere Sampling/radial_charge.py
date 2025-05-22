# 단위 구와 이를 감싸는 큰 구가 존재합니다.
# 큰 구의 한 점에서 브라운 운동을 시작할 경우, 단위 구에 도달할 확률은 1/r에 비례합니다.
# 본 프로그램은 큰 구 내에서 무작위로 선택된 점에서 출발한 입자가 단위 구 표면에
# 처음 도달하는 위치의 확률 분포를 계산합니다.
# 절차:
# 1. 큰 구에서 N개의 점을 균일하게 샘플링합니다.
# 2. 각 점에 대해 단위 구에 도달할 확률을 평가합니다.
# 3. 입자가 단위 구 표면에 최초 도달한 위치의 확률 분포를 산출합니다.
# 4. 결과적으로, 큰 구의 임의의 점에서 출발하여 단위 구의 특정 점 Y에 도달하는 확률 밀도 함수를 구합니다.
# 입력: 큰 구의 반지름 R, 단위 구의 반지름 r

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from bin import Bin  # Import Bin class from the bin module
np.random.seed(42)  # For reproducibility

def sample_points_on_sphere(radius, num_points):
    """Generate random points on the surface of a sphere."""
    points = []
    for _ in range(num_points):
        z = radius * np.random.uniform(-1, 1)
        theta = np.acos(z / radius)
        phi = np.random.uniform(0, 2 * np.pi)
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        points.append((x, y, z))
    assert np.all(np.abs(np.linalg.norm(np.array(points), axis=1) - radius) < 1e-8)
    return np.array(points)

def get_reach_prob(point):
    """Calculate the probability of reaching the **unit** sphere from a point."""

    distance = np.linalg.norm(point)
    if distance <= 1 - 1e-8:
        raise ValueError("Point is inside the unit sphere")
    return 1 / distance

import numpy as np

def _rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """
    두 3-D 벡터 `from_vec` → `to_vec` 로 정렬시키는 3×3 회전행렬을 계산한다.
    ── (Rodrigues' rotation formula)
    """
    # Create working copies to avoid modifying original inputs and ensure float type
    a_w = np.array(from_vec, dtype=float, copy=True)
    b_w = np.array(to_vec, dtype=float, copy=True)

    # Calculate norms
    norm_a = np.linalg.norm(a_w)
    norm_b = np.linalg.norm(b_w)

    # Check for zero vectors
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Input vectors must be non-zero for rotation.")

    # 단위벡터화 (Normalize working copies)
    a_w /= norm_a
    b_w /= norm_b

    v = np.cross(a_w, b_w)        # 회전축 (정규화 X)
    s = np.linalg.norm(v)         # sin θ
    if s == 0:                    # 이미 평행(±) → 항등 또는 180°
        # Check dot product of normalized vectors to determine direction
        return np.eye(3) if np.dot(a_w, b_w) > 0 else -np.eye(3)

    c = np.dot(a_w, b_w)          # cos θ
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    # Rodrigues' formula
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    return R


def rotate_like(ref_from: np.ndarray,
                ref_to: np.ndarray,
                vecs: np.ndarray) -> np.ndarray:
    """
    ref_from  : shape (3,)
    ref_to    : shape (3,)
    vecs      : shape (..., 3) - (3,), (n, 3), (b₁, b₂, …, 3) 모두 허용

    반환값    : 입력과 동일한 leading shape (..., 3)
    """
    vecs = np.asarray(vecs, dtype=float)
    R = _rotation_matrix(ref_from, ref_to)         # 3×3
    # 마지막 차원(=3)을 Rᵀ와 곱한다 → 브로드캐스트로 전 배치 처리
    return vecs @ R.T

def sigma(theta, p, a = 1, q = 1):
    """
    전도성 구 표면에서의 전하 밀도 계산.
    여기서는 random walker의 first passage를 구하기 위해 사용.
    점전하는 (0, 0, p)에 있다고 가정

    Parameters:
        a : float
            구의 반지름
        theta : float or np.ndarray
            극각 (rad 단위)
        q : float
            점전하의 크기
        p : float
            점전하의 위치 (구 중심으로부터의 거리)

    Returns:
        float or np.ndarray
            표면 전하 밀도 sigma(a, theta)
    """
    if p <= a - 1e-8:
        raise ValueError("The point charge must be outside the sphere (p > a).")
    # TODO: 이 식 검토해야함
    numerator   = q * (p**2 - a**2)
    denominator = 4 * np.pi * a * (p**2 - 2 * a * p * np.cos(theta) + a**2)**1.5
    return numerator / denominator


def get_distribution(center_offset_z, point) -> Callable[[np.ndarray], np.ndarray]:
    """
    Calculate the distribution of the point where the random walker
    first reaches the off-centered unit sphere.
    Assume the center of the sphere is at (0, 0, center_offset_z).
    Use the theorem in the electrostatic problem.

    Parameters:
        center_offset_z : float
            구의 중심에서 z축으로의 오프셋
        point : np.ndarray, shape (3,)
            랜덤 워크를 시작하는 점

    return: function
        distribution에서 점을 샘플링하는 함수
    """
    # Adjust the coordinate
    # Then the center of the sphere is at (0, 0, 0)
    point = point - np.array([0, 0, center_offset_z])
    # Rotate the point to the z-axis
    point_rotated = rotate_like(point, (0, 0, 1), point)
    # Calculate the distance from the point to the center of the sphere
    distance = np.linalg.norm(point_rotated)
    if distance <= 1 - 1e-8:
        raise ValueError("Point is inside the unit sphere")

    def get_prob_density(x: np.ndarray) -> np.ndarray:
        """Calculate the probability density at points on the unit sphere.

        Args:
            x: np.ndarray, shape (3,) or (n, 3)
                Points on the unit sphere surface.

        Returns:
            np.ndarray: The probability density at points x.
                If x.shape == (3,), returns float.
                If x.shape == (n, 3), returns array of shape (n,).
        """
        # Reshape to (n, 3) if single point
        if x.ndim == 1:
            x = x.reshape(1, 3)
            single_point = True
        else:
            single_point = False

        # Adjust the coordinate to the unit sphere
        x = x - np.array([0, 0, center_offset_z])
        # assert x is on the unit sphere
        assert np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-8) , \
            f"x is not on the unit sphere: {np.max(np.abs(np.linalg.norm(x, axis=1) - 1))}"
        # rotate x according to starting point of random walker
        x_rotated = rotate_like(point, point_rotated, x)
        theta = np.acos(np.clip(x_rotated[:, 2], -1, 1))

        # get the charge density at the points.
        # The charge density is the prob density of the points on the sphere.
        charge_density = sigma(theta=theta, p=distance, a=1, q=1)

        return charge_density[0] if single_point else charge_density

    return get_prob_density

def main():
    z_offset = 0
    n_bin = 100
    large_radius = 10
    n_sample_point = 10**4
    sample_per_point = 10**3

    bin = Bin(r=1, bin_nums=n_bin)
    # 큰 구에서 무작위로 점을 샘플링
    sampled_points = sample_points_on_sphere(large_radius, n_sample_point)
    for i, point in enumerate(sampled_points):
        # 샘플링된 점에서 시작한 랜덤 워커가 단위 구에 도달할 확률
        reach_prob = get_reach_prob(point)
        # 단위구 표면에서 랜덤 워커가 단위 구에 도달할 확률 분포
        # get_distribution은 샘플링된 모든 점이 단위 구에 도달한다고 가정함.
        prob_density = get_distribution(z_offset, point)

        # prob_density를 반영하여 랜덤 워커의 first passage 샘플링 개수를 조정
        n_samples = int(sample_per_point * reach_prob)
        # 점을 균일하게 샘플링하고 확률 밀도를 곱하여 랜덤 워커의 first passage 분포를 샘플링
        sample_points = sample_points_on_sphere(1, n_samples)
        densities = prob_density(sample_points)
        fp_densities = sample_points * densities[:, np.newaxis]  # Reshape densities to (n, 1) for broadcasting
        # 샘플링된 first passage 분포를 빈에 추가
        bin.count(fp_densities)
        print(f"{i}th iteration is done")
    bin.visualize(mode="deviation", show_deviation=True)
    bin.save("radial_charge.npz")

if __name__ == "__main__":
    main()