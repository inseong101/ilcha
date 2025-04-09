import os
import numpy as np
import osmnx as ox
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")  # 설치 후 사용 가능

# ✅ 폰트를 Arial로 설정 (한글 사용 방지)
mpl.rc('font', family='Arial')

# ✅ Mac에서 matplotlib 백엔드 설정 (필요 시)
import matplotlib
matplotlib.use("Qt5Agg")  # Mac에서 GUI 백엔드 문제 해결

# ✅ 경로 설정
base_path = "/Supplementary_coding_for_everyone/4번_유사성분석"
north_arrow_path = os.path.join(base_path, "다운로드.jpeg")
save_path = os.path.join(base_path, "KMC_MC_NHI_Distribution")

# ✅ 대전시 도로망 & 경계 데이터 가져오기
place_name = "Daejeon, South Korea"
print("🔄 도로망 데이터 불러오는 중...")
G = ox.graph_from_place(place_name, network_type='drive', simplify=True)
G = ox.project_graph(G, to_crs="EPSG:3857")  # ✅ 도로망 좌표 변환

# ✅ 대전 경계 데이터 가져오기
print("🔄 대전 경계 데이터 불러오는 중...")
gdf_boundary = ox.geocode_to_gdf(place_name).to_crs(epsg=3857)

# ✅ 지도 설정 (오른쪽 & 아래 여백 추가)
fig, ax = plt.subplots(figsize=(12, 9))
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.95)  # 오른쪽 & 아래 여백 조정

# ✅ 지도 배경 추가
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=3857, zoom=12)

# ✅ 대전 경계선 추가
gdf_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=3, zorder=500)

# ✅ 도로망 추가
ox.plot_graph(G, ax=ax, node_size=0, edge_color="black", edge_alpha=0.1, edge_linewidth=0.5, show=False, close=False)

# ✅ 의료기관 데이터 불러오기
file_path_medical = os.path.join(base_path, "processed_hospitals_updated.xlsx")
file_path_bogun = os.path.join(base_path, "processed_Bogun_updated.xlsx")

df_medical = pd.read_excel(file_path_medical)
df_bogun = pd.read_excel(file_path_bogun)

# ✅ 대전시 의료기관 필터링
df_medical_daejeon = df_medical[df_medical["시도"] == "대전광역시"]
df_bogun_daejeon = df_bogun[df_bogun["시도"] == "대전광역시"]

# ✅ 의료기관 좌표 변환 (EPSG:3857)
gdf_medical = gpd.GeoDataFrame(df_medical_daejeon, geometry=gpd.points_from_xy(df_medical_daejeon["경도"], df_medical_daejeon["위도"]), crs="EPSG:4326")
gdf_bogun = gpd.GeoDataFrame(df_bogun_daejeon, geometry=gpd.points_from_xy(df_bogun_daejeon["경도"], df_bogun_daejeon["위도"]), crs="EPSG:4326")

gdf_medical = gdf_medical.to_crs(epsg=3857)
gdf_bogun = gdf_bogun.to_crs(epsg=3857)

mc_coords = gdf_medical[gdf_medical["분류"] == "의원"]["geometry"].apply(lambda p: (p.x, p.y)).tolist()
kmc_coords = gdf_medical[gdf_medical["분류"] == "한의원"]["geometry"].apply(lambda p: (p.x, p.y)).tolist()
nhi_coords = gdf_bogun["geometry"].apply(lambda p: (p.x, p.y)).tolist()

# ✅ 의료기관 위치 점 추가
ax.scatter(*zip(*mc_coords), c='red', label='MC', s=8, alpha=0.2)
# ax.scatter(*zip(*kmc_coords), c='blue', label='KMC', s=8, alpha=0.8)
# ax.scatter(*zip(*nhi_coords), c='green', label='NHI', s=8, alpha=0.8)
#
# # ✅ 범례 추가
# ax.legend(frameon=True, facecolor='white', edgecolor='black', loc='upper left', fontsize=12)

# ✅ 현재 x, y 범위 확인
x_min, y_min, x_max, y_max = gdf_boundary.total_bounds
x_range = x_max - x_min
y_range = y_max - y_min

# ✅ 여백 추가 (오른쪽 & 아래쪽 확장)
padding_x = x_range * 0.2  # x축 패딩 비율 (10%)
padding_y = y_range * 0.2  # y축 패딩 비율 (10%)

new_x_min = x_min - x_range * 0.05  # 왼쪽 확장
new_x_max = x_max + padding_x  # 오른쪽 확장
new_y_min = y_min - y_range * 0.05  # 아래 확장
new_y_max = y_max + padding_y  # 위 확장

# ✅ **도로망을 그리기 전에 x, y 범위를 미리 설정**
ax.set_xlim(new_x_min, new_x_max)
ax.set_ylim(new_y_min, new_y_max)

# ✅ 방위 화살표 추가 (오른쪽 위)
north_arrow = Image.open(north_arrow_path).convert("RGBA")
extent = [
    x_max - x_range * 0.12, x_max - x_range * 0.03,  # X 범위
    y_max - y_range * 0.12, y_max - y_range * 0.03   # Y 범위
]
ax.imshow(north_arrow, extent=extent, alpha=1, zorder=500)

# ✅ 축척 추가 (오른쪽 아래로 이동)
def add_scalebar(ax, x_start, y_start, length_km=10):
    """0-5-10km 축척 추가"""
    bar_height = y_range * 0.02  # 축척 바 높이 조정
    scale_factor = (new_x_max - new_x_min) / x_range
    segment_length = length_km * 500 * scale_factor  # 5km 간격 (미터 단위)

    # 큰 바 추가
    ax.add_patch(patches.Rectangle((x_start, y_start), segment_length * 2, bar_height,
                                   edgecolor='black', facecolor='white', linewidth=2, zorder=254))
    ax.add_patch(patches.Rectangle((x_start + segment_length, y_start), segment_length, bar_height,
                                   edgecolor='black', facecolor='gray', linewidth=2, zorder=254))

    # 작은 1km 간격 선 추가
    for i in range(1, 10):
        x_pos = x_start + (i * (segment_length // 5))
        ax.plot([x_pos, x_pos], [y_start, y_start + bar_height // 2], color='black', linewidth=2, zorder=254)

    # 거리 텍스트 추가
    ax.text(x_start, y_start - bar_height * 1.5, '0', fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='center', zorder=254)
    ax.text(x_start + segment_length, y_start - bar_height * 1.5, '5 km', fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='center', zorder=254)
    ax.text(x_start + segment_length * 2, y_start - bar_height * 1.5, '10 km', fontsize=12, fontweight='bold',
            verticalalignment='top', horizontalalignment='center', zorder=254)

# ✅ 축척을 오른쪽 아래로 이동
add_scalebar(ax, x_max - x_range * 0.3, y_min, length_km=10)

# ✅ X, Y 축 삭제
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
ax.set_frame_on(False)

# ✅ 그래프 저장 (오른쪽 & 아래 여백 추가)
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, "mc_distribution_fixed.png")
plt.savefig(output_path, format="png", dpi=2000, bbox_inches="tight", pad_inches=0.1)  # ✅ 여백 확대
plt.show()

print(f"✅ 지도 저장 완료! 저장 경로: {output_path}")


