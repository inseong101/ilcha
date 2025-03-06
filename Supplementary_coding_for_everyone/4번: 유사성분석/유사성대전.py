import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

# ✅ 로컬 환경에서 파일 경로 설정
base_path = "/Users/iinseong/Desktop/ilcha_clean/Supplementary_coding_for_everyone/4번: 유사성분석"  # 파일이 저장된 폴더 경로
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
base_path = "/Users/iinseong/Desktop/ilcha_clean/Supplementary_coding_for_everyone/4번: 유사성분석/"  # 파일이 저장된 폴더 경로
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

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

# ✅ 로컬 환경에서 파일 경로 설정
base_path = "/Users/iinseong/Desktop/ilcha_clean/Supplementary_coding_for_everyone/4번: 유사성분석/"  # 파일이 저장된 폴더 경로
dist_matrix_path = os.path.join(base_path, "dist_matrix.npy")
nodes_path = os.path.join(base_path, "nodes.npy")
save_path = os.path.join(base_path, "Cross_K_Function_Analysis")

# ✅ 저장 폴더 생성
os.makedirs(save_path, exist_ok=True)

# ✅ `dist_matrix` 및 `nodes` 불러오기
if os.path.exists(dist_matrix_path) and os.path.exists(nodes_path):
    print("\n✅ 저장된 최단 거리 행렬 로드 중...")
    nodes = np.load(nodes_path, allow_pickle=True).tolist()
    dist_matrix = np.load(dist_matrix_path)
    print("✅ 최단 거리 행렬 로드 완료!")
else:
    raise FileNotFoundError("\n❌ 저장된 `dist_matrix.npy` 또는 `nodes.npy` 파일이 없습니다. 경로를 확인하세요!")

# ✅ 거리 범위 설정
r_values = np.linspace(0, 30000, 50)


def compute_cross_k_fast(nodes, dist_matrix, sample_nodes1, sample_nodes2, distances):
    idx1 = [nodes.index(n) for n in sample_nodes1]
    idx2 = [nodes.index(n) for n in sample_nodes2]
    dist_submatrix = dist_matrix[idx1][:, idx2].copy()
    k_values = []
    for d in tqdm(distances, desc="Cross-K 계산 중"):
        count_per_point = np.zeros(len(sample_nodes1))
        for k in range(len(sample_nodes1)):
            count_per_point[k] = np.count_nonzero(dist_submatrix[k, :] <= d)
        k_values.append(np.sum(count_per_point) / len(sample_nodes1))
    return np.array(k_values)


# ✅ Cross-K Function 실행
obs_k_mc_nhi = compute_cross_k_fast(nodes, dist_matrix, nhi_nodes, mc_nodes, r_values)
obs_k_kmc_nhi = compute_cross_k_fast(nodes, dist_matrix,  nhi_nodes, kmc_nodes, r_values)
obs_k_kmc_mc = compute_cross_k_fast(nodes, dist_matrix,  mc_nodes, kmc_nodes, r_values)


@jit(nopython=True, parallel=True)
def monte_carlo_cross_k_fast(dist_matrix, sample_size1, sample_size2, distances, num_simulations=100):
    n = dist_matrix.shape[0]
    csr_k_values = np.zeros((num_simulations, len(distances)))
    for i in range(num_simulations):
        sample_idx1 = np.random.choice(n, sample_size1, replace=False)
        sample_idx2 = np.random.choice(n, sample_size2, replace=False)
        dist_submatrix = dist_matrix[sample_idx1][:, sample_idx2]
        for j, d in enumerate(distances):
            csr_k_values[i, j] = np.sum(dist_submatrix <= d) / sample_size1
    return csr_k_values


num_simulations = 100
csr_k_mc_nhi = monte_carlo_cross_k_fast(dist_matrix,  len(nhi_nodes), len(mc_nodes), r_values, num_simulations)
csr_k_kmc_nhi = monte_carlo_cross_k_fast(dist_matrix, len(nhi_nodes), len(kmc_nodes),  r_values, num_simulations)
csr_k_kmc_mc = monte_carlo_cross_k_fast(dist_matrix, len(kmc_nodes), len(mc_nodes), r_values, num_simulations)


def compute_csr_bounds(csr_k_values):
    return np.mean(csr_k_values, axis=0), np.percentile(csr_k_values, 97.5, axis=0), np.percentile(csr_k_values, 2.5,
                                                                                                   axis=0)


csr_mean_mc_nhi, csr_upper_mc_nhi, csr_lower_mc_nhi = compute_csr_bounds(csr_k_mc_nhi)
csr_mean_kmc_nhi, csr_upper_kmc_nhi, csr_lower_kmc_nhi = compute_csr_bounds(csr_k_kmc_nhi)
csr_mean_kmc_mc, csr_upper_kmc_mc, csr_lower_kmc_mc = compute_csr_bounds(csr_k_kmc_mc)

# ✅ 개별 그래프 저장 함수
def save_cross_k_plot(obs_k, csr_mean, csr_upper, csr_lower, label, filename):
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
save_cross_k_plot(obs_k_mc_nhi, csr_mean_mc_nhi, csr_upper_mc_nhi, csr_lower_mc_nhi, "MC vs. NHI", "cross_k_mc_nhi.png")
save_cross_k_plot(obs_k_kmc_nhi, csr_mean_kmc_nhi, csr_upper_kmc_nhi, csr_lower_kmc_nhi, "KMC vs. NHI", "cross_k_kmc_nhi.png")
save_cross_k_plot(obs_k_kmc_mc, csr_mean_kmc_mc, csr_upper_kmc_mc, csr_lower_kmc_mc, "KMC vs. MC", "cross_k_kmc_mc.png")

print(f"✅ 그래프 저장 완료! 저장 경로: {save_path}")






import osmnx as ox
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib_scalebar.scalebar import ScaleBar

# ✅ 로컬 환경에서 파일 경로 설정
base_path = "/Users/iinseong/Desktop/ilcha_clean/Supplementary_coding_for_everyone/4번: 유사성분석"  # 파일이 저장된 폴더 경로
dist_matrix_path = os.path.join(base_path, "dist_matrix.npy")
nodes_path = os.path.join(base_path, "nodes.npy")
north_arrow_path = os.path.join(base_path, "다운로드.png")  # 북쪽 방향 PNG 파일
save_path = os.path.join(base_path, "KMC_MC_NHI_Distribution")

# ✅ 저장 폴더 생성
os.makedirs(save_path, exist_ok=True)

# ✅ `dist_matrix` 및 `nodes` 불러오기
if os.path.exists(dist_matrix_path) and os.path.exists(nodes_path):
    print("\n✅ 저장된 최단 거리 행렬 로드 중...")
    nodes = np.load(nodes_path, allow_pickle=True).tolist()
    dist_matrix = np.load(dist_matrix_path)
    print("✅ 최단 거리 행렬 로드 완료!")
else:
    raise FileNotFoundError("\n❌ 저장된 `dist_matrix.npy` 또는 `nodes.npy` 파일이 없습니다. 경로를 확인하세요!")

# ✅ 거리 범위 설정
r_values = np.linspace(0, 30000, 50)

# ✅ 대전시 도로망 가져오기 (네트워크)
place_name = "Daejeon, South Korea"
G = ox.graph_from_place(place_name, network_type='drive')

# ✅ 대전 경계선 (Boundary) 가져오기
gdf_boundary = ox.geocode_to_gdf(place_name)

# ✅ 의료기관 위치 변환 (도로망 노드 좌표 사용)
mc_coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in mc_nodes]
kmc_coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in kmc_nodes]
nhi_coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in nhi_nodes]

# ✅ 좌표계를 미터 단위로 변환 (EPSG:5186 or UTM)
gdf_boundary = gdf_boundary.to_crs(epsg=5186)

# ✅ 지도 설정
fig, ax = plt.subplots(figsize=(8, 8))

# ✅ 지도 배경 추가 (위성 or 지형도)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf_boundary.crs)

# ✅ 대전 경계선 추가
gdf_boundary.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1)

# ✅ 도로망 추가
ox.plot_graph(G, ax=ax, node_size=0, edge_color="gray", edge_alpha=0.3, show=False, close=False)

# ✅ 의료기관 위치 점 찍기 (dot 사이즈 조정, NHI도 dot으로 변경)
ax.scatter(*zip(*mc_coords), c='red', label='MC (의원)', s=5, alpha=0.8)  # MC: 빨간색
ax.scatter(*zip(*kmc_coords), c='blue', label='KMC (한의원)', s=5, alpha=0.8)  # KMC: 파란색
ax.scatter(*zip(*nhi_coords), c='green', label='NHI (보건소)', s=5, alpha=0.8)  # NHI: 초록색 (dot으로 변경)

# ✅ 북쪽 방향 PNG 이미지 추가
north_arrow = mpimg.imread(north_arrow_path)
ax.imshow(north_arrow, aspect='auto', extent=[0.85, 0.95, 0.85, 0.95], transform=ax.transAxes)

# ✅ 축척 추가 (지도 거리 반영)
scalebar = ScaleBar(1, location='lower left', units='km', scale_loc='bottom',
                    length_fraction=0.2, scale_bar_style='line', label_style='plain',
                    font_properties={'size': 10}, dimension="si-length", fixed_value=1000)
ax.add_artist(scalebar)

# ✅ 레이블 삭제 (제목 제거)
ax.legend(frameon=False, loc='upper right', fontsize=8)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# ✅ 그래프 저장
output_path = os.path.join(save_path, "kmc_mc_nhi_distribution.png")
plt.savefig(output_path, dpi=2000, bbox_inches="tight")
plt.show()

print(f"✅ 지도 저장 완료! 저장 경로: {output_path}")