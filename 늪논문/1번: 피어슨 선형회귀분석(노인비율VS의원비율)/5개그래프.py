import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
plt.rcParams['font.family'] = 'AppleGothic'

# 📌 파일 로드
file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path)

# 노인 비율 데이터 로드
age_population_file = "전처리된_노인비율.xlsx"
df_age = pd.read_excel(age_population_file)
df_age["시군구_통합"] = df_age["행정기관"].str.strip()

# 📌 일차의료 만성질환관리 시범사업 참여의원 파일 로드
file_path_chronic_care = "정확히_일치하는_의원목록.xlsx"  # ✅ 매칭된 파일 사용
df_chronic_care = pd.read_excel(file_path_chronic_care)

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

# 📌 데이터 병합
df_med_ratio = pd.merge(hanmed_ratio, clinic_ratio, on="시군구_통합", how="outer").fillna(0)

df_med_ratio["한의원 비율"] = df_med_ratio["한의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"])

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
df_internal_medicine_sum.rename(columns={"가중치_의료인수": "내과의원 의료인수 총합"}, inplace=True)

df_family_medicine_sum = df_family_medicine.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
df_family_medicine_sum.rename(columns={"가중치_의료인수": "가정의학과의원 의료인수 총합"}, inplace=True)

df_non_specialized_sum = df_non_specialized.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
df_non_specialized_sum.rename(columns={"가중치_의료인수": "미표방 의원 의료인수 총합"}, inplace=True)

# 📌 병합
df_med_ratio = pd.merge(df_med_ratio, df_internal_medicine_sum, on="시군구_통합", how="left").fillna(0)
df_med_ratio = pd.merge(df_med_ratio, df_family_medicine_sum, on="시군구_통합", how="left").fillna(0)
df_med_ratio = pd.merge(df_med_ratio, df_non_specialized_sum, on="시군구_통합", how="left").fillna(0)

# ✅ 시범사업 참여 의원 의료인수 총합 계산 (수정된 코드)
matched_df = df_medical[df_medical["사업장명"].isin(df_chronic_care["요양기관명"])].copy()

# ✅ "시군구_통합" 컬럼 추가 (없어서 발생한 KeyError 해결!)
matched_df["시군구_통합"] = (matched_df["시도"] + " " + matched_df["시군구"]).str.strip()

# ✅ 의료인수 총합 계산
df_chronic_sum = matched_df.groupby("시군구_통합")["의료인수"].sum().reset_index()
df_chronic_sum.rename(columns={"의료인수": "시범사업 참여의원 의료인수 총합"}, inplace=True)

# ✅ 병합
df_med_ratio = pd.merge(df_med_ratio, df_chronic_sum, on="시군구_통합", how="left").fillna(0)

# ✅ "시군구_통합"을 기준으로 병합
df_med_ratio = pd.merge(df_med_ratio, df_age[["시군구_통합", "노인 비율"]], on="시군구_통합", how="left")

# ✅ 병합 후 "노인 비율"이 존재하는지 확인
print(df_med_ratio.columns)  # 🔍 "노인 비율"이 있는지 확인
print(df_med_ratio[["시군구_통합", "노인 비율"]].head())  # 🔍 값이 채워졌는지 확인

# 📌 ✅ 비율 계산
df_med_ratio["내과의원 비율"] = df_med_ratio["내과의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"])
df_med_ratio["가정의학과의원 비율"] = df_med_ratio["가정의학과의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"])
df_med_ratio["미표방 의원 비율"] = df_med_ratio["미표방 의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"])
df_med_ratio["시범사업 참여의원 비율"] = df_med_ratio["시범사업 참여의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"])

# 📌 피어슨 상관 분석
correlation_results = [(col, *pearsonr(df_med_ratio["노인 비율"], df_med_ratio[col])) for col in
                       ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율"]]
df_correlation = pd.DataFrame(correlation_results, columns=["변수", "피어슨 상관계수 (r)", "p-value"])
df_correlation["샘플 크기 (n)"] = len(df_med_ratio)
print(df_correlation)

# 📌 5개 그래프 그리기
plt.figure(figsize=(10, 6))
for col in ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율"]:
    sns.regplot(x=df_med_ratio["노인 비율"], y=df_med_ratio[col], label=col, scatter_kws={'alpha':0.5})

plt.legend()
plt.xlabel("노인 비율")
plt.ylabel("의료기관 비율")
plt.title("노인 비율과 의료기관 유형 간 관계")
plt.savefig("5개의그림.png", dpi=300, bbox_inches='tight')
plt.show()