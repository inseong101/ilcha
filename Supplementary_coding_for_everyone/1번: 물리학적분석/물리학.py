import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # Mac (Windows는 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# 파일 경로
population_file = "전처리된_노인비율+면적.xlsx"
bogun_file = "processed_Bogun_updated.xlsx"
hospitals_file = "processed_hospitals_updated.xlsx"

# 데이터 불러오기
population_df = pd.read_excel(population_file)
bogun_df = pd.read_excel(bogun_file)
hospitals_df = pd.read_excel(hospitals_file)

# 데이터 불러오기
population_df = pd.read_excel(population_file)
bogun_df = pd.read_excel(bogun_file)
hospitals_df = pd.read_excel(hospitals_file)

# 컬럼명 확인
print("컬럼 확인 (Population):", population_df.columns)
print("컬럼 확인 (Bogun):", bogun_df.columns)
print("컬럼 확인 (Hospitals):", hospitals_df.columns)

# '지역구' 컬럼 생성 ('시도' + '시군구' 결합, 단 세종시는 '시도'만 사용)
bogun_df["지역구"] = bogun_df["시도"] + " " + bogun_df["시군구"].fillna("")
hospitals_df["지역구"] = hospitals_df["시도"] + " " + hospitals_df["시군구"].fillna("")

# '행정기관'을 '지역구'로 변경
population_df.rename(columns={'행정기관': '지역구'}, inplace=True)

# 의원과 한의원을 분리
clinic_df = hospitals_df[hospitals_df['분류'] == '의원']
oriental_df = hospitals_df[hospitals_df['분류'] == '한의원']
all_df = hospitals_df[(hospitals_df['분류'] == '한의원') | (hospitals_df['분류'] == '의원')]


# Step 1: 지역별 시설 개수 계산
def count_facilities_by_region(facility_df):
    return facility_df.groupby("지역구").size().reset_index(name='시설 수')


bogun_count = count_facilities_by_region(bogun_df)
clinic_count = count_facilities_by_region(clinic_df)
oriental_count = count_facilities_by_region(oriental_df)
all_count = count_facilities_by_region(all_df)

# Step 2: 지역별 인구수와 시설 수 합치기
merged_bogun = pd.merge(population_df, bogun_count, on="지역구", how="left").fillna(0)
merged_clinic = pd.merge(population_df, clinic_count, on="지역구", how="left").fillna(0)
merged_oriental = pd.merge(population_df, oriental_count, on="지역구", how="left").fillna(0)
merged_all = pd.merge(population_df, all_count, on="지역구", how="left").fillna(0)


# Step 3: 인구 밀도와 시설 밀도 계산
def calculate_density(df, pop_col="총 인구수", noin_col="노인 비율", facility_col="시설 수", area_col="면적"):
    df["인구 밀도"] = df[pop_col] / df[area_col]
    df["시설 밀도"] = df[facility_col] / df[area_col]
    return df


merged_bogun = calculate_density(merged_bogun)
merged_clinic = calculate_density(merged_clinic)
merged_oriental = calculate_density(merged_oriental)
merged_all = calculate_density(merged_all)


# Step 4: 로그 변환 및 그래프 분석
def log_log_analysis(df, title):
    df = df[(df["인구 밀도"] > 0) & (df["시설 밀도"] > 0)]
    log_p = np.log10(df["인구 밀도"])
    log_D = np.log10(df["시설 밀도"])

    slope, intercept, r_value, p_value, std_err = linregress(log_p, log_D)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=log_p, y=log_D)
    plt.plot(log_p, slope * log_p + intercept, color='red', label=f'α = {slope:.3f}')
    plt.xlabel("log 인구 밀도")
    plt.ylabel("log 시설 밀도")
    plt.title(title + f' (α = {slope:.3f})')
    plt.legend()
    plt.show()

    return slope


# Step 5: 로그-로그 회귀 분석 수행
alpha_bogun = log_log_analysis(merged_bogun, "보건소 배치 패턴")
alpha_clinic = log_log_analysis(merged_clinic, "의원 배치 패턴")
alpha_oriental = log_log_analysis(merged_oriental, "한의원 배치 패턴")
alpha_all = log_log_analysis(merged_all, "Primary clinic")

# Step 6: 결과 해석
print(f"보건소의 α 값: {alpha_bogun}")
print(f"의원의 α 값: {alpha_clinic}")
print(f"한의원의 α 값: {alpha_oriental}")
print(f"Primary clinic의 α 값: {alpha_all}")

if alpha_bogun < 1:
    print("보건소는 공공시설적인 배치를 보입니다 (α < 1).")
else:
    print("보건소는 상업시설적인 배치를 보입니다 (α > 1).")

if alpha_clinic < 1:
    print("의원은 공공시설적인 배치를 보입니다 (α < 1).")
else:
    print("의원은 상업시설적인 배치를 보입니다 (α > 1).")

if alpha_oriental < 1:
    print("한의원은 공공시설적인 배치를 보입니다 (α < 1).")
else:
    print("한의원은 상업시설적인 배치를 보입니다 (α > 1).")