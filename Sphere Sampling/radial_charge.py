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

import numpy as np
np.random.seed(42)  # For reproducibility

def sample_points_on_sphere(radius, num_points):
    """Generate random points on the surface of a sphere."""
    points = []
    for _ in range(num_points):
        z = radius * np.random.uniform(-1, 1)
        theta = np.acos(z / radius)
        phi = np.random.uniform(0, 2 * np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        points.append((x, y, z))
    return np.array(points)

def get_reach_prob(point):
    """Calculate the probability of reaching the **unit** sphere from a point."""
    distance = np.linalg.norm(point)
    if distance <= 1:
        raise ValueError("Point is inside the unit sphere")
    return 1 / distance

def rotate_to_z(point: np.ndarray):
    """
    Rotate the point to the z-axis.
    Return the value of z
    """
    sign = np.sign(point[2])
    z = np.linalg.norm(point)
    return z if sign > 0 else -z

import numpy as np

def rotate_like(x: np.ndarray,
                x_prime: np.ndarray,
                v: np.ndarray,
                eps: float = 1e-8) -> np.ndarray:
    """
    Return `v` rotated by the minimal rotation that aligns `x` with `x_prime`.

    Parameters:
        x, x_prime : array_like, shape (3,)
            원래 벡터와 회전 후 목표 방향 벡터.
        v : array_like, shape (3,)
            위 회전( x → x′ )을 적용할 벡터.
        eps : float, optional
            직교성·평행성 판단에 사용할 허용 오차.

    Returns:
        np.ndarray, shape (3,)
            회전된 v.
    """
    x   = np.asarray(x, dtype=float)
    xp  = np.asarray(x_prime, dtype=float)
    v   = np.asarray(v, dtype=float)

    # 1. 단위벡터화
    if (nx := np.linalg.norm(x))  < eps or (nxp := np.linalg.norm(xp)) < eps:
        raise ValueError("x, x_prime must be non-zero vectors.")
    u = x  / nx
    w = xp / nxp

    # 2. 회전축(axis)·각도(theta)
    axis = np.cross(u, w)
    axis_len = np.linalg.norm(axis)
    dot      = np.clip(np.dot(u, w), -1.0, 1.0)          # 수치 안정화

    # 2-a. x ‖ x′ : 축·각도 특수 처리
    if axis_len < eps:
        if dot > 0:                                      # 이미 정렬
            return v.copy()
        # 180° 회전 ─ x와 수직인 임의 축 선택
        axis = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < eps:
            axis = np.cross(u, [0.0, 1.0, 0.0])
        axis_len = np.linalg.norm(axis)

    axis /= axis_len
    theta = np.arccos(dot)

    # 3. Rodrigues 공식으로 v 회전
    k      = axis
    kv     = np.dot(k, v)
    cos_t  = np.cos(theta)
    sin_t  = np.sin(theta)

    v_rot = (v * cos_t
             + np.cross(k, v) * sin_t
             + k * kv * (1.0 - cos_t))
    return v_rot

def sigma(theta, p, a = 1, q = 1):
    """
    전도성 구 표면에서의 전하 밀도 계산.
    여기서는 random walker의 first passage를 구하기 위해 사용.

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
    numerator = q * (a**2 - p**2)
    denominator = 4 * np.pi * a * (a**2 - 2 * a * p * np.cos(theta) + p**2)**(1.5)
    return numerator / denominator


def get_fp_sampler(center_offset_z, point):
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
    # Calculate the distance from the point to the center of the sphere
    distance = np.linalg.norm(point)
    if distance <= 1:
        raise ValueError("Point is inside the unit sphere")

    def sampling(x):
        """Calculate the distribution of the point on the sphere."""
        # Rotate the point to the z-axis
        z = rotate_to_z(x)
        theta = np.acos(np.random.uniform(-1, 1))
        # sample the point on the sphere using the therorem on charge density
        sampled_point = sigma(theta, z, 1, 1)
        # Rotate the point to the original coordinate
        result = rotate_like((0, 0, z), x, sampled_point)
        # Adjust the coordinate
        result = result + np.array([0, 0, center_offset_z])
        return result

    return sampling

# TODO bin을 설정해서 실제로 샘플링하기(main.py를 만들어서 하는게 좋을듯)