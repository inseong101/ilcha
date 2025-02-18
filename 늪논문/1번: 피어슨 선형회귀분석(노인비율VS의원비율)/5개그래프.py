
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📌 파일 로드
file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path)

# 📌 노인 비율 데이터 로드
age_population_file = "전처리된_노인비율.xlsx"
df_age = pd.read_excel(age_population_file)
df_age["시군구_통합"] = df_age["행정기관"].str.strip()

# 📌 한의원 및 의원 데이터 필터링
df_hanmed = df_medical[df_medical["분류"] == "한의원"].copy()
df_clinic = df_medical[df_medical["분류"] == "의원"].copy()

# 📌 가중치(의료인수) 적용
df_hanmed["가중치_의료인수"] = df_hanmed["의료인수"]
df_clinic["가중치_의료인수"] = df_clinic["의료인수"]

# 📌 시군구 통합 컬럼 생성
df_hanmed["시군구_통합"] = (df_hanmed["시도"] + " " + df_hanmed["시군구"]).str.strip()
df_clinic["시군구_통합"] = (df_clinic["시도"] + " " + df_clinic["시군구"]).str.strip()

# 📌 한의원 및 의원 의료인수 총합 계산
hanmed_ratio = df_hanmed.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
clinic_ratio = df_clinic.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()

# 📌 컬럼명 변경
hanmed_ratio.rename(columns={"가중치_의료인수": "한의원 의료인수 총합"}, inplace=True)
clinic_ratio.rename(columns={"가중치_의료인수": "의원 의료인수 총합"}, inplace=True)

# 📌 기본 데이터 병합
df_med_ratio = pd.merge(hanmed_ratio, clinic_ratio, on="시군구_통합", how="outer").fillna(0)

# 📌 전체 의료기관 의료인수 계산 (한의원 + 의원)
df_med_ratio["전체 의료기관 의료인수"] = df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"]

# 📌 한의원 비율
df_med_ratio["한의원 비율"] = df_med_ratio["한의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]

# ✅ '노인 비율' 데이터 병합
df_med_ratio = pd.merge(df_med_ratio, df_age[["시군구_통합", "노인 비율"]], on="시군구_통합", how="left")

# 📌 내과의원, 가정의학과의원, 미표방 의원 필터링
df_internal_medicine = df_clinic[df_clinic["사업장명"].str.endswith("내과의원")]
df_family_medicine = df_clinic[df_clinic["사업장명"].str.endswith("가정의학과의원")]

specialties = ["내과", "신경과", "정신건강의학과", "정신과", "외과", "정형외과", "신경외과", "심장혈관흉부외과",
               "성형외과", "마취통증의학과", "마취과", "산부인과", "소아청소년과", "소아과", "안과", "이비인후과",
               "피부과", "비뇨의학과", "비뇨기과", "영상의학과", "방사선종양학과", "병리과", "진단검사의학과", "재활의학과",
               "결핵과", "예방의학과", "가정의학과", "핵의학과", "직업환경의학과", "응급의학과"]

pattern = "|".join([f"{sp}의원$" for sp in specialties])
df_non_specialized = df_clinic[~df_clinic["사업장명"].str.contains(pattern)]

# 📌 의료인수 총합 계산
df_internal_medicine_sum = df_internal_medicine.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
df_family_medicine_sum = df_family_medicine.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
df_non_specialized_sum = df_non_specialized.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()

df_internal_medicine_sum.rename(columns={"가중치_의료인수": "내과의원 의료인수 총합"}, inplace=True)
df_family_medicine_sum.rename(columns={"가중치_의료인수": "가정의학과의원 의료인수 총합"}, inplace=True)
df_non_specialized_sum.rename(columns={"가중치_의료인수": "미표방 의원 의료인수 총합"}, inplace=True)

# ✅ 시범사업 참여 의원 데이터 불러오기
df_chronic_care = pd.read_excel("정확히_일치하는_의원목록.xlsx")

# ✅ 'processed_hospitals_updated.xlsx'에서 정확히 일치하는 의원 찾기
import numpy as np

# ✅ '요양기관명'을 numpy 배열이 아니라 set으로 변환
chronic_care_set = set(df_chronic_care["요양기관명"])

# ✅ 정확하게 일치하는 값만 필터링
df_chronic_clinic = df_medical[df_medical["사업장명"].apply(lambda x: x in chronic_care_set)].copy()
print(f"✅ 정확히 일치하는 의원 개수: {df_chronic_clinic.shape[0]}")
print(df_chronic_care["요양기관명"].value_counts().head(10))  # 상위 10개 중복 확인

# ✅ '시군구_통합' 컬럼 추가
df_chronic_clinic["시군구_통합"] = (df_chronic_clinic["시도"] + " " + df_chronic_clinic["시군구"]).str.strip()

# ✅ 의료인수 총합 계산 (시군구별 그룹화)
df_chronic_sum = df_chronic_clinic.groupby("시군구_통합")["의료인수"].sum().reset_index()
df_chronic_sum.rename(columns={"의료인수": "시범사업 참여의원 의료인수 총합"}, inplace=True)

# ✅ 모든 데이터 병합
df_med_ratio = pd.merge(df_med_ratio, df_internal_medicine_sum, on="시군구_통합", how="left").fillna(0)
df_med_ratio = pd.merge(df_med_ratio, df_family_medicine_sum, on="시군구_통합", how="left").fillna(0)
df_med_ratio = pd.merge(df_med_ratio, df_non_specialized_sum, on="시군구_통합", how="left").fillna(0)
df_med_ratio = pd.merge(df_med_ratio, df_chronic_sum, on="시군구_통합", how="left").fillna(0)

# ✅ 비율 계산 (Y축을 "의료기관 비율"로 조정)
df_med_ratio["내과의원 비율"] = df_med_ratio["내과의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]
df_med_ratio["가정의학과의원 비율"] = df_med_ratio["가정의학과의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]
df_med_ratio["미표방 의원 비율"] = df_med_ratio["미표방 의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]
df_med_ratio["시범사업 참여의원 비율"] = df_med_ratio["시범사업 참여의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]

# ✅ 피어슨 상관 분석 및 시각화
plt.figure(figsize=(10, 6))
for col in ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율"]:
    sns.regplot(x=df_med_ratio["노인 비율"], y=df_med_ratio[col], label=col, scatter_kws={'alpha': 0.5})

print("🔍 df_med_ratio 컬럼 목록:", df_med_ratio.columns)
print(df_med_ratio.head())  # 상위 5개 행 출력하여 값 확인

# ✅ 피어슨 상관 분석
correlation_results = [(col, *pearsonr(df_med_ratio["노인 비율"], df_med_ratio[col])) for col in
                       ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율"]]
df_correlation = pd.DataFrame(correlation_results, columns=["변수", "피어슨 상관계수 (r)", "p-value"])

# ✅ 각 의료기관 유형별 개수(n) 및 총 의료인 수(nn) 계산
num_hanmed = df_hanmed.shape[0]  # 한의원 개수
num_internal_medicine = df_internal_medicine.shape[0]  # 내과의원 개수
num_family_medicine = df_family_medicine.shape[0]  # 가정의학과의원 개수
num_non_specialized = df_non_specialized.shape[0]  # 미표방 의원 개수
num_chronic_care = df_chronic_clinic.shape[0]  # 시범사업 참여의원 개수

# ✅ 총 의료인 수 계산
sum_hanmed = df_hanmed["의료인수"].sum()  # 한의원 총 의료인수
sum_internal_medicine = df_internal_medicine["의료인수"].sum()  # 내과의원 총 의료인수
sum_family_medicine = df_family_medicine["의료인수"].sum()  # 가정의학과의원 총 의료인수
sum_non_specialized = df_non_specialized["의료인수"].sum()  # 미표방 의원 총 의료인수
sum_chronic_care = df_chronic_clinic["의료인수"].sum()  # 시범사업 참여의원 총 의료인수

# ✅ 개수(n) 및 총 의료인수(nn) 리스트 생성
num_values = [num_hanmed, num_internal_medicine, num_family_medicine, num_non_specialized, num_chronic_care]
num_medical_staff_values = [sum_hanmed, sum_internal_medicine, sum_family_medicine, sum_non_specialized, sum_chronic_care]

# ✅ df_correlation에 개수(n) 및 의료인수(nn) 추가
df_correlation["의료기관 개수 (n)"] = num_values
df_correlation["총 의료인 수 (nn)"] = num_medical_staff_values

# ✅ 최종 결과 출력
print(df_correlation)

plt.legend()
plt.xlabel("노인 비율")
plt.ylabel("의료기관 비율")
plt.title("노인 비율과 의료기관 유형 간 관계")
plt.savefig("5개의그림.png", dpi=300, bbox_inches='tight')
plt.show()


# ✅ 피어슨 상관계수 막대그래프
plt.figure(figsize=(8, 5))
sns.barplot(
    x="변수", y="피어슨 상관계수 (r)", hue="변수", data=df_correlation, palette="coolwarm", legend=False
)

# ✅ 그래프 꾸미기
plt.axhline(0, color="black", linewidth=1)  # 0 기준선 추가
plt.xticks()
plt.ylabel("피어슨 상관계수 (r)")
plt.title("노인 비율과 의료기관 유형 간 피어슨 상관계수 비교")

# ✅ 그래프 저장
plt.savefig("피어슨_막대그래프.png", dpi=300, bbox_inches='tight')

# ✅ 그래프 표시
plt.show()