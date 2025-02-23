import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

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


# ✅ KDE 밀도를 직접 계산하는 함수
def compute_kde(data, grid_size=100):
    """ KDE를 계산하고 격자로 변환 """
    if len(data) < 2:  # 데이터 부족 시 예외 처리
        return np.zeros(grid_size * grid_size)

    x, y = data["경도"], data["위도"]
    values = np.vstack([x, y])

    kde = gaussian_kde(values)  # KDE 계산
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    density = kde(positions).reshape(xx.shape)
    return density.flatten()


# ✅ Bhattacharyya Distance 계산 함수
def bhattacharyya_distance(p, q):
    """두 KDE 분포 간 바타차리야 거리 계산"""
    p = p / np.sum(p)  # 정규화
    q = q / np.sum(q)  # 정규화
    return -np.log(np.sum(np.sqrt(p * q)))


# ✅ KDE 히트맵을 그리는 함수
def plot_kde(data, title, ax):
    x, y = data["경도"], data["위도"]
    if len(x) > 1:  # 데이터가 충분한 경우 KDE 계산
        sns.kdeplot(x=x, y=y, ax=ax, cmap="Reds", fill=True, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")


# ✅ KDE 히트맵 플롯 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_kde(bogun_coords, "보건소 KDE 히트맵", axes[0])
plot_kde(clinic_coords, "의원 KDE 히트맵", axes[1])
plot_kde(han_clinic_coords, "한의원 KDE 히트맵", axes[2])

plt.tight_layout()
plt.savefig("KDE.png", dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import pandas as pd

# 📌 파일 로드
file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path)

Bogun_file_path = "processed_Bogun_updated.xlsx"
df_Bogun = pd.read_excel(Bogun_file_path)

# 📌 한의원 및 의원 데이터 필터링
df_hanmed = df_medical[df_medical["분류"] == "한의원"].copy()
df_clinic = df_medical[df_medical["분류"] == "의원"].copy()

# 📌 내과의원, 가정의학과의원, 미표방 의원 필터링
df_internal_medicine = df_clinic[df_clinic["사업장명"].str.endswith("내과의원")]
df_family_medicine = df_clinic[df_clinic["사업장명"].str.endswith("가정의학과의원")]

specialties = ["내과", "신경과", "정신건강의학과", "정신과", "외과", "정형외과", "신경외과", "심장혈관흉부외과",
               "성형외과", "마취통증의학과", "마취과", "산부인과", "소아청소년과", "소아과", "안과", "이비인후과",
               "피부과", "비뇨의학과", "비뇨기과", "영상의학과", "방사선종양학과", "병리과", "진단검사의학과", "재활의학과",
               "결핵과", "예방의학과", "가정의학과", "핵의학과", "직업환경의학과", "응급의학과"]

pattern = "|".join([f"{sp}의원$" for sp in specialties])
df_non_specialized = df_clinic[~df_clinic["사업장명"].str.contains(pattern)]



import pandas as pd

# ✅ 국민건강보험공단 엑셀파일 불러오기
file_path_chronic_care = "국민건강보험공단_일차의료 만성질환관리 시범사업 참여의원 목록_20240331.csv"
df_chronic_care = pd.read_csv(file_path_chronic_care, encoding="cp949")

# ✅ processed_hospitals_updated.xlsx 불러오기
file_path_hospitals = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path_hospitals)

# ✅ 우편번호 5자리 변환 (4자리인 경우 앞에 0 추가)
df_chronic_care["우편번호"] = df_chronic_care["우편번호"].astype(str).str.strip().str.zfill(5)

# ✅ 컬럼 정리 및 데이터 전처리 (공백 제거)
df_chronic_care["요양기관명"] = df_chronic_care["요양기관명"].str.strip()
df_chronic_care["우편번호"] = df_chronic_care["우편번호"].astype(str).str.strip()

df_medical["사업장명"] = df_medical["사업장명"].str.strip()
df_medical["도로명우편번호"] = df_medical["도로명우편번호"].astype(str).str.strip()

# ✅ 정확한 일치 조건: '사업장명 == 요양기관명' & '도로명우편번호 == 우편번호'
df_chronic_clinic = df_medical.merge(df_chronic_care,
                                     left_on=["사업장명", "도로명우편번호"],
                                     right_on=["요양기관명", "우편번호"],
                                     how="inner")

# ✅ 필요 없는 컬럼 제거 (중복 제거 후)
df_chronic_clinic.drop(columns=["요양기관명", "우편번호"], inplace=True)

# ✅ '시군구_통합' 컬럼 추가
df_chronic_clinic["시군구_통합"] = (df_chronic_clinic["시도"] + " " + df_chronic_clinic["시군구"]).str.strip()

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 보건소 위경도 데이터 추출
bogun_coords = df_Bogun[["경도", "위도"]].dropna()

# ✅ 각 의료기관 유형별 위경도 데이터 추출
clinic_category_coords = {
    "한의원": df_hanmed[["경도", "위도"]].dropna(),
    "내과의원": df_internal_medicine[["경도", "위도"]].dropna(),
    "가정의학과의원": df_family_medicine[["경도", "위도"]].dropna(),
    "미표방의원": df_non_specialized[["경도", "위도"]].dropna(),
    "시범사업 참여의원": df_chronic_clinic[["경도", "위도"]].dropna(),
}

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import pandas as pd

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac은 AppleGothic, Windows는 Malgun Gothic 가능
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ✅ 보건소 위경도 데이터 추출
bogun_coords = df_Bogun[["경도", "위도"]].dropna()

# ✅ 각 의료기관 유형별 위경도 데이터 추출
clinic_category_coords = {
    "한의원": df_hanmed[["경도", "위도"]].dropna(),
    "내과의원": df_internal_medicine[["경도", "위도"]].dropna(),
    "가정의학과의원": df_family_medicine[["경도", "위도"]].dropna(),
    "미표방의원": df_non_specialized[["경도", "위도"]].dropna(),
    "시범사업 참여의원": df_chronic_clinic[["경도", "위도"]].dropna(),
}

# ✅ 각 의료기관 유형별 개수 확인
clinic_counts = {category: len(coords) for category, coords in clinic_category_coords.items()}
clinic_counts["보건소"] = len(bogun_coords)  # 보건소 개수 추가

# ✅ 개수 출력
print("🔹 의료기관 유형별 개소 수")
for key, value in clinic_counts.items():
    print(f"{key}: {value}개")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde

# ✅ KDE 계산 함수
def compute_kde(data, grid_size=100):
    """ KDE를 계산하고 격자로 변환 """
    if len(data) < 2:  # 데이터 부족 시 예외 처리
        return np.zeros(grid_size * grid_size)

    x, y = data["경도"], data["위도"]
    values = np.vstack([x, y])

    kde = gaussian_kde(values)  # KDE 계산
    x_grid = np.linspace(x.min(), x.max(), grid_size)
    y_grid = np.linspace(y.min(), y.max(), grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    density = kde(positions).reshape(xx.shape)
    return density.flatten()

# ✅ Bhattacharyya Distance 계산 함수
def bhattacharyya_distance(p, q):
    """ 두 KDE 분포 간 바타차리야 거리 계산 """
    p = p / np.sum(p)  # 정규화
    q = q / np.sum(q)  # 정규화
    return -np.log(np.sum(np.sqrt(p * q)))

# ✅ 보건소 및 의료기관 유형별 데이터 저장
clinic_category_coords = {
    "보건소": bogun_coords,
    "의원": clinic_coords,
    "한의원": df_hanmed[["경도", "위도"]].dropna(),
    "내과의원": df_internal_medicine[["경도", "위도"]].dropna(),
    "가정의학과의원": df_family_medicine[["경도", "위도"]].dropna(),
    "미표방의원": df_non_specialized[["경도", "위도"]].dropna(),
    "시범사업 참여의원": df_chronic_clinic[["경도", "위도"]].dropna(),
}

# ✅ 모든 의료기관 간 바타차리야 거리 계산 (7x7 행렬)
categories = list(clinic_category_coords.keys())
num_categories = len(categories)

# ✅ 거리 행렬 초기화
distance_matrix = np.zeros((num_categories, num_categories))

# ✅ 모든 유형 간 바타차리야 거리 계산
kde_dict = {category: compute_kde(coords) for category, coords in clinic_category_coords.items() if len(coords) > 1}

for i in range(num_categories):
    for j in range(i, num_categories):  # 대칭 행렬이므로 절반만 계산
        if i == j:
            distance_matrix[i, j] = 0  # 동일한 분포 간 거리는 0
        else:
            dist = bhattacharyya_distance(kde_dict[categories[i]], kde_dict[categories[j]])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # 대칭 적용

# ✅ DataFrame으로 변환
df_distance_matrix = pd.DataFrame(distance_matrix, index=categories, columns=categories)

# ✅ 결과 출력
print("\n🔹 의료기관 유형 간 Bhattacharyya Distance 행렬")
print(df_distance_matrix)

# ✅ 거리 행렬 시각화 (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(df_distance_matrix, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
plt.title("의료기관 유형 간 Bhattacharyya Distance 행렬")
plt.ylabel("기관 유형")
plt.xlabel("기관 유형")
plt.savefig("바타차리야_거리_행렬.png", dpi=300, bbox_inches='tight')
plt.show()