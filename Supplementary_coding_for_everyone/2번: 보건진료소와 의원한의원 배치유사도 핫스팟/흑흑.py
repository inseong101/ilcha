import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from geopy.distance import great_circle
import matplotlib.pyplot as plt
from tqdm import tqdm

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac (Windows는 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ✅ 파일 로드
file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path)

Bogun_file_path = "processed_Bogun_updated.xlsx"
df_Bogun = pd.read_excel(Bogun_file_path)

# ✅ 의료기관 유형별 좌표 추출
clinic_category_coords = {
    "보건소": df_Bogun[["경도", "위도"]].dropna(),
    "의원": df_medical[df_medical["분류"] == "의원"][["경도", "위도"]].dropna(),
    "한의원": df_medical[df_medical["분류"] == "한의원"][["경도", "위도"]].dropna(),
}

# ✅ 연구 지역 면적 보정 (예: 50,000 km²)
area_km2 = 100210


# ✅ Haversine 변환을 적용한 최근접 이웃 거리(NND) 계산
def compute_nnd(coords):
    """ Haversine 변환을 적용한 최근접 이웃 거리(NND) 계산 (위도-경도 순서 수정 및 tqdm 진행 상황 표시) """
    if len(coords) < 2:
        return np.nan  # 데이터 부족 예외 처리

    coords_array = coords.to_numpy()  # NumPy 배열 변환
    num_points = len(coords_array)

    nearest_distances = []

    for i in tqdm(range(num_points), desc="최근접 이웃 거리 계산"):
        # ✅ 위경도 순서 수정 (위도, 경도)
        lat_lon_i = coords_array[i][::-1]  # (경도, 위도) → (위도, 경도)로 변환

        dists = [
            great_circle(lat_lon_i, coords_array[j][::-1]).km  # (위도, 경도)로 변환하여 계산
            for j in range(num_points) if i != j
        ]

        nearest_distances.append(min(dists))  # 최근접 거리 저장

    return np.mean(nearest_distances)  # 평균 NND 반환


# ✅ 각 의료기관 유형별 NNR 계산
nnr_results = {}

for category, coords in clinic_category_coords.items():
    print(f"\n🔹 {category} NNR 계산 중...")
    n_o = compute_nnd(coords)  # 실제 평균 최근접 이웃 거리
    n_e = 0.5 / np.sqrt(len(coords) / area_km2)  # 기대 평균 거리

    if np.isnan(n_o) or n_e == 0:  # 데이터 부족 예외 처리
        nnr = np.nan
    else:
        nnr = n_o / n_e  # NNR 계산

    nnr_results[category] = {"NND(실제)": n_o, "NND(기대)": n_e, "NNR": nnr}

# ✅ 결과 출력
nnr_df = pd.DataFrame.from_dict(nnr_results, orient="index")
print("\n🔹 수정된 의료기관 유형별 NNR 분석 결과")
print(nnr_df)

# ✅ 결과 시각화
plt.figure(figsize=(8, 6))
nnr_df["NNR"].plot(kind="bar", color=["red", "blue", "green"], alpha=0.7)
plt.axhline(y=1, color="black", linestyle="--", label="무작위 분포 기준 (NNR=1)")
plt.ylabel("NNR 값")
plt.title("의료기관 유형별 NNR (수정된 계산)")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("NNR_수정된_분석.png", dpi=300, bbox_inches="tight")
plt.show()