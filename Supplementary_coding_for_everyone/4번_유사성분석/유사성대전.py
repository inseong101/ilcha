import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

# ✅ 로컬 환경에서 파일 경로 설정
base_path = "/Users/iinseong/Desktop/ilcha_clean/Supplementary_coding_for_everyone/4번_유사성분석/"  # 파일이 저장된 폴더 경로
file_path_medical = os.path.join(base_path, "processed_hospitals_updated.xlsx")
file_path_bogun = os.path.join(base_path, "processed_Bogun_updated.xlsx")

dist_matrix_path = os.path.join(base_path, "dist_matrix.npy")
nodes_path = os.path.join(base_path, "nodes.npy")

# ✅ 1. 의료기관 데이터 불러오기
df_medical = pd.read_excel(file_path_medical)
df_bogun = pd.read_excel(file_path_bogun)

# ✅ 2. 대전시 의료기관만 필터링
df_medical_daejeon = df_medical[df_medical["시도"] == "대전광역시"]
df_bogun_daejeon = df_bogun[df_bogun["시도"] == "대전광역시"]

# ✅ 3. 의료기관별 데이터 분리
mc_daejeon = df_medical_daejeon[df_medical_daejeon["분류"] == "의원"]
kmc_daejeon = df_medical_daejeon[df_medical_daejeon["분류"] == "한의원"]
nhi_daejeon = df_bogun_daejeon

# ✅ 4. 의료기관 개수 확인
print("대전시 의원 개수:", len(mc_daejeon))
print("대전시 한의원 개수:", len(kmc_daejeon))
print("대전시 보건소 개수:", len(nhi_daejeon))

# ✅ 5. 대전시 도로망 가져오기
place_name = "Daejeon, South Korea"
G = ox.graph_from_place(place_name, network_type='drive')

# ✅ 6. 의료기관을 도로망의 최근접 노드에 매칭
def match_nodes(G, df):
    return df.apply(lambda row: ox.distance.nearest_nodes(G, row["경도"], row["위도"]), axis=1).tolist()

mc_nodes = match_nodes(G, mc_daejeon)
kmc_nodes = match_nodes(G, kmc_daejeon)
nhi_nodes = match_nodes(G, nhi_daejeon)

# ✅ 네트워크 내 모든 노드 간 최단 거리 행렬 계산
def compute_distance_matrix(G):
    nodes = list(G.nodes)
    dist_matrix = np.full((len(nodes), len(nodes)), np.inf)
    for i, node1 in enumerate(tqdm(nodes, desc="최단 거리 행렬 계산 중")):
        lengths = nx.single_source_dijkstra_path_length(G, node1, weight='length')
        for j, node2 in enumerate(nodes):
            if node2 in lengths:
                dist_matrix[i, j] = lengths[node2]
    return nodes, dist_matrix

# ✅ 최단 거리 행렬 계산 및 저장
nodes, dist_matrix = compute_distance_matrix(G)
np.save(nodes_path, np.array(nodes, dtype=object))
np.save(dist_matrix_path, dist_matrix)

print("✅ 최단 거리 행렬 및 노드 리스트 저장 완료!")

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

# ✅ 로컬 환경에서 파일 경로 설정
base_path = "/Supplementary_coding_for_everyone/4번_유사성분석/"  # 파일이 저장된 폴더 경로
dist_matrix_path = os.path.join(base_path, "dist_matrix.npy")
nodes_path = os.path.join(base_path, "nodes.npy")
save_path = os.path.join(base_path, "Auto_K_Function_Analysis")

# ✅ 저장 폴더 생성
os.makedirs(save_path, exist_ok=True)

# ✅ `dist_matrix` 및 `nodes` 불러오기 (파일이 존재하면 로드)
if os.path.exists(dist_matrix_path) and os.path.exists(nodes_path):
    print("\n✅ 저장된 최단 거리 행렬 로드 중...")
    nodes = np.load(nodes_path, allow_pickle=True).tolist()  # 리스트 형태로 변환
    dist_matrix = np.load(dist_matrix_path)
    print("✅ 최단 거리 행렬 로드 완료!")
else:
    raise FileNotFoundError("\n❌ 저장된 `dist_matrix.npy` 또는 `nodes.npy` 파일이 없습니다. 경로를 확인하세요!")

# ✅ 거리 범위 설정
r_values = np.linspace(0, 30000, 50)


def compute_auto_k_fast(nodes, dist_matrix, sample_nodes, distances):
    idx = [nodes.index(n) for n in sample_nodes]
    dist_submatrix = dist_matrix[idx][:, idx].copy()
    np.fill_diagonal(dist_submatrix, np.inf)
    k_values = []
    N = len(sample_nodes)
    for d in tqdm(distances, desc="Auto-K 계산 중"):
        count_per_point = np.zeros(len(sample_nodes))
        for k in range(len(sample_nodes)):
            count_per_point[k] = np.count_nonzero(dist_submatrix[k, :] <= d)
        k_values.append(np.sum(count_per_point) / N)
    return np.array(k_values)


# ✅ Auto-K Function 실행
obs_k_mc = compute_auto_k_fast(nodes, dist_matrix, mc_nodes, r_values)
obs_k_kmc = compute_auto_k_fast(nodes, dist_matrix, kmc_nodes, r_values)
obs_k_nhi = compute_auto_k_fast(nodes, dist_matrix, nhi_nodes, r_values)


@jit(nopython=True, parallel=True)
def monte_carlo_simulation_fast(dist_matrix, sample_size, distances, num_simulations=100):
    n = dist_matrix.shape[0]
    csr_k_values = np.zeros((num_simulations, len(distances)))
    for i in range(num_simulations):
        sample_idx = np.random.choice(n, sample_size, replace=False)
        dist_submatrix = dist_matrix[sample_idx][:, sample_idx].copy()
        np.fill_diagonal(dist_submatrix, np.inf)
        for j, d in enumerate(distances):
            count_per_point = np.zeros(sample_size)
            for k in range(sample_size):
                count_per_point[k] = np.count_nonzero(dist_submatrix[k, :] <= d)
            csr_k_values[i, j] = np.sum(count_per_point) / sample_size
    return csr_k_values


num_simulations = 100
csr_k_mc = monte_carlo_simulation_fast(dist_matrix, len(mc_nodes), r_values, num_simulations)
csr_k_kmc = monte_carlo_simulation_fast(dist_matrix, len(kmc_nodes), r_values, num_simulations)
csr_k_nhi = monte_carlo_simulation_fast(dist_matrix, len(nhi_nodes), r_values, num_simulations)


def compute_csr_bounds(csr_k_values):
    return np.mean(csr_k_values, axis=0), np.percentile(csr_k_values, 97.5, axis=0), np.percentile(csr_k_values, 2.5,
                                                                                                   axis=0)


csr_mean_mc, csr_upper_mc, csr_lower_mc = compute_csr_bounds(csr_k_mc)
csr_mean_kmc, csr_upper_kmc, csr_lower_kmc = compute_csr_bounds(csr_k_kmc)
csr_mean_nhi, csr_upper_nhi, csr_lower_nhi = compute_csr_bounds(csr_k_nhi)

# ✅ 개별 그래프 저장 함수
def save_auto_k_plot(obs_k, csr_mean, csr_upper, csr_lower, label, filename):
    plt.figure(figsize=(7, 5))
    plt.fill_between(r_values, csr_lower, csr_upper, color="gray", alpha=0.3)
    plt.plot(r_values, csr_mean, label="Exp(Mean)", linestyle="-", color="green")
    plt.plot(r_values, obs_k, label="Obs", linestyle="-", color="blue")
    plt.xlabel("Distance (m)")
    plt.ylabel("Cumulative number of points")
    plt.legend()
    plt.savefig(os.path.join(save_path, filename), dpi=500, bbox_inches="tight")
    plt.close()

# ✅ 그래프 저장 실행
save_auto_k_plot(obs_k_mc, csr_mean_mc, csr_upper_mc, csr_lower_mc, "MC", "auto_k_mc.png")
save_auto_k_plot(obs_k_kmc, csr_mean_kmc, csr_upper_kmc, csr_lower_kmc, "KMC", "auto_k_kmc.png")
save_auto_k_plot(obs_k_nhi, csr_mean_nhi, csr_upper_nhi, csr_lower_nhi, "NHI", "auto_k_nhi.png")

print(f"✅ 그래프 저장 완료! 저장 경로: {save_path}")
plt.tight_layout()
plt.show()

print(f"✅ 그래프 저장 완료! 저장 경로: {save_path}")