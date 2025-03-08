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
    "보건소 (NHI)": df_Bogun[["경도", "위도"]].dropna(),
    "의원 (MC)": df_medical[df_medical["분류"] == "의원"][["경도", "위도"]].dropna(),
    "한의원 (KMC)": df_medical[df_medical["분류"] == "한의원"][["경도", "위도"]].dropna(),
}

# ✅ 연구 지역 면적 (예: km² 단위)
area_km2 = 100210

# ✅ Haversine 변환을 적용한 최근접 이웃 거리(NND) 계산
def compute_nnd(coords):
    """ Haversine 변환을 적용한 최근접 이웃 거리(NND) 계산 """
    if len(coords) < 2:
        return np.nan  # 데이터 부족 예외 처리

    coords_array = coords.to_numpy()
    num_points = len(coords_array)

    nearest_distances = []

    for i in tqdm(range(num_points), desc="최근접 이웃 거리 계산"):
        lat_lon_i = coords_array[i][::-1]  # (경도, 위도) → (위도, 경도)

        dists = [
            great_circle(lat_lon_i, coords_array[j][::-1]).km
            for j in range(num_points) if i != j
        ]

        nearest_distances.append(min(dists))  # 최근접 거리 저장

    return np.mean(nearest_distances)  # 평균 NND 반환


# ✅ 각 의료기관 유형별 ANNR 계산
annr_results = {}

for category, coords in clinic_category_coords.items():
    print(f"\n🔹 {category} ANNR 계산 중...")
    d_obs = compute_nnd(coords)  # 실제 평균 최근접 이웃 거리
    d_exp = 0.5 / np.sqrt(len(coords) / area_km2)  # 기대 평균 거리

    if np.isnan(d_obs) or d_exp == 0:  # 데이터 부족 예외 처리
        annr = np.nan
        z_score = np.nan
    else:
        annr = d_obs / d_exp  # ANNR 계산
        std_error = 0.26136 / np.sqrt(len(coords) / area_km2)  # 표준 오차 계산
        z_score = (d_obs - d_exp) / std_error  # Z-score 계산

    # ✅ 패턴 판별 (Clustered, Random, Dispersed)
    if annr < 1:
        pattern = "Clustered"
    elif annr > 1:
        pattern = "Dispersed"
    else:
        pattern = "Random"

    annr_results[category] = {
        "Expected Distance (m)": round(d_exp * 1000, 4),  # km → m 변환
        "Observed Distance (m)": round(d_obs * 1000, 4),  # km → m 변환
        "ANN Ratio": round(annr, 4),
        "Z-score": round(z_score, 6),
        "Pattern": pattern
    }

# ✅ 결과를 표 형태로 출력
annr_df = pd.DataFrame.from_dict(annr_results, orient="index")

print("\n🔹 의료기관 유형별 ANNR 분석 결과")
print(annr_df)

# ✅ 결과 저장
annr_df.to_csv("ANNR_분석_결과.csv", encoding="utf-8-sig")

# ✅ 결과 시각화
plt.figure(figsize=(8, 6))
annr_df["ANN Ratio"].plot(kind="bar", color=["red", "blue", "green"], alpha=0.7)
plt.axhline(y=1, color="black", linestyle="--", label="무작위 분포 기준 (ANNR=1)")
plt.ylabel("ANN Ratio 값")
plt.title("의료기관 유형별 ANNR")
plt.xticks(rotation=45)
plt.legend()
plt.savefig("ANNR_분석_결과.png", dpi=300, bbox_inches="tight")
plt.show()