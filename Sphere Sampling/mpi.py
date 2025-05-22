from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from bin import Bin
from radial_charge import sample_points_on_sphere, get_reach_prob, get_distribution

def main():
    # MPI 설정
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"rank: {rank}, size: {size}")

    # 파라미터 설정 (radial_charge.py와 동일하게 유지)
    z_offset = 0
    n_bin = 100
    large_radius = 10
    n_total_sample_point = 10**8  # 전체 샘플 포인트 수
    sample_per_point = 10**3

    # 작업 분배: 각 프로세스가 전체 샘플 포인트 중 일부를 처리
    points_per_process = n_total_sample_point // size
    remainder_points = n_total_sample_point % size

    # 각 프로세스가 처리할 샘플 포인트 수 계산
    local_n_sample_point = points_per_process
    if rank < remainder_points:
        local_n_sample_point += 1

    if rank == 0:
        print(f"Total sample points: {n_total_sample_point}")
        print(f"Number of processes: {size}")
        print(f"Points per process: ~{points_per_process}")

    # 각 프로세스는 자체 Bin 객체를 가짐
    local_bin = Bin(r=1, bin_nums=n_bin)

    # 각 프로세스마다 다른 시드 설정 (선택사항)
    np.random.seed(42 + rank)

    # 각 프로세스는 자신의 샘플 포인트를 생성
    sampled_points_local = sample_points_on_sphere(large_radius, local_n_sample_point)

    for i, point in enumerate(sampled_points_local):
        # 단위 구에 도달할 확률 계산
        reach_prob = get_reach_prob(point)

        # 확률 밀도 함수 가져오기
        prob_density_func = get_distribution(z_offset, point)

        # 샘플링 개수 조정
        n_samples_fp = int(sample_per_point * reach_prob)
        if n_samples_fp == 0:
            continue

        # first passage 분포 샘플링
        fp_sample_points = sample_points_on_sphere(1, n_samples_fp)
        densities = prob_density_func(fp_sample_points)

        # densities 형태 조정 (broadcasting 준비)
        if densities.ndim == 0 and n_samples_fp == 1:  # 단일 포인트일 경우
            densities = np.array([densities])
        elif densities.ndim > 1:
            densities = densities.flatten()

        # 가중치 적용
        fp_densities = fp_sample_points * densities[:, np.newaxis]

        # 로컬 빈에 추가
        local_bin.count(fp_densities)

        # 진행 상황 출력 (랭크 0만)
        if rank == 0 and (i + 1) % max(1, local_n_sample_point // 10) == 0:
            print(f"Rank 0: {i+1}/{local_n_sample_point} points processed ({(i+1)/local_n_sample_point*100:.1f}%)")

    # 결과 집계 - 모든 프로세스의 bins를 합침
    if rank == 0:
        print("모든 로컬 계산 완료. 결과 집계 중...")
        global_bins = np.copy(local_bin.bins)  # 랭크 0의 결과로 시작

        # 다른 모든 프로세스로부터 bins 받아오기
        for i in range(1, size):
            received_bins = comm.recv(source=i, tag=11)
            global_bins += received_bins

        # 최종 Bin 객체 생성 및 시각화
        final_bin = Bin(r=1, bin_nums=n_bin)
        final_bin.bins = global_bins
        print("결과 집계 완료. 시각화 중...")
        final_bin.visualize(mode="deviation", show_deviation=True)
        final_bin.save("radial_charge_mpi.npz")
        print("결과가 radial_charge_mpi.npz로 저장되었습니다.")
    else:
        # 랭크 0이 아닌 프로세스는 자신의 bins를 랭크 0으로 전송
        comm.send(local_bin.bins, dest=0, tag=11)
        print(f"Rank {rank}: 계산 완료, 데이터 전송 완료.")

if __name__ == "__main__":
    main()