import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac (Windows는 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ✅ 데이터 불러오기 (파일 경로 수정)
bogun_file = "processed_Bogun_updated.xlsx"
hospitals_file = "processed_hospitals_updated.xlsx"

bogun_df = pd.read_excel(bogun_file)
hospitals_df = pd.read_excel(hospitals_file)

# ✅ 보건소, 의원, 한의원 데이터 분리
bogun_coords = bogun_df[["경도", "위도"]].dropna()
clinic_coords = hospitals_df[hospitals_df["분류"] == "의원"][["경도", "위도"]].dropna()
han_clinic_coords = hospitals_df[hospitals_df["분류"] == "한의원"][["경도", "위도"]].dropna()


# ✅ KDE 히트맵을 그리는 함수
def plot_kde(data, title, ax):
    x, y = data["경도"], data["위도"]
    if len(x) > 1:  # 데이터가 충분한 경우 KDE 계산
        sns.kdeplot(x=x, y=y, ax=ax, cmap="Reds", fill=True, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")


# ✅ Bhattacharyya Distance 계산 함수
def bhattacharyya_distance(p, q):
    """두 KDE 분포 간 바타차리야 거리 계산"""
    return -np.log(np.sum(np.sqrt(p * q)))


# ✅ KDE 히스토그램으로 변환하는 함수
def compute_kde(data, grid_size=100):
    """ KDE 히트맵을 생성하고, 격자로 변환 """
    x, y = data["경도"], data["위도"]
    kde = sns.kdeplot(x=x, y=y, fill=True, cmap="Reds", alpha=0.5)

    # 히스토그램 데이터를 격자로 변환
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # KDE 확률 밀도 추정
    density = np.exp(kde.get_lines()[0].get_data()[1])  # KDE 값 추출
    return density.flatten()


# ✅ KDE 히트맵 플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_kde(bogun_coords, "보건소 KDE 히트맵", axes[0])
plot_kde(clinic_coords, "의원 KDE 히트맵", axes[1])
plot_kde(han_clinic_coords, "한의원 KDE 히트맵", axes[2])

plt.tight_layout()
plt.show()

# ✅ Bhattacharyya Distance 계산
bogun_kde = compute_kde(bogun_coords)
clinic_kde = compute_kde(clinic_coords)
han_clinic_kde = compute_kde(han_clinic_coords)

bhatt_clinic = bhattacharyya_distance(bogun_kde, clinic_kde)
bhatt_han_clinic = bhattacharyya_distance(bogun_kde, han_clinic_kde)

# ✅ 결과 출력
print(f"🔹 Bhattacharyya Distance (보건소 vs. 의원): {bhatt_clinic:.4f}")
print(f"🔹 Bhattacharyya Distance (보건소 vs. 한의원): {bhatt_han_clinic:.4f}")

# ✅ 그래프 생성 (분포 유사성 비교)
df_results = pd.DataFrame({
    "비교 대상": ["보건소 vs. 의원", "보건소 vs. 한의원"],
    "Bhattacharyya Distance": [bhatt_clinic, bhatt_han_clinic]
})

df_results.set_index("비교 대상").plot(kind="bar", figsize=(8, 5), colormap="coolwarm", edgecolor="black")
plt.title("보건소 vs. 의원 & 한의원의 공간적 유사성 비교")
plt.ylabel("Bhattacharyya Distance (낮을수록 유사)")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()