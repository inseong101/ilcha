import pyproj
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1️⃣ 엑셀 파일 로드
file_path = "지방행정인허가에서_다운받은_파일.xlsx"
#다운받는 곳: https://www.localdata.go.kr/devcenter/dataDown.do?menuNo=20001 여기서 맨 밑에 의원 xlsx로 다운받으세요.
df = pd.read_excel(file_path, engine="openpyxl")

# 2️⃣ "의원" 또는 "한의원"에 해당하고, "영업상태명"이 "영업/정상"인 데이터 필터링
filtered_df = df[
    (df["의료기관종별명"].isin(["보건소", "보건지소", "보건진료소"])) & (df["영업상태명"] == "영업/정상")
].copy()

# 3️⃣ EPSG:5174 -> EPSG:4326 (위도/경도) 변환
transformer = pyproj.Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)

# 좌표 변환 적용
filtered_df[["경도", "위도"]] = filtered_df.apply(
    lambda row: pd.Series(transformer.transform(row["좌표정보X(EPSG5174)"], row["좌표정보Y(EPSG5174)"])),
    axis=1
)

# 4️⃣ 필요한 컬럼만 유지
filtered_df = filtered_df[[
    "업태구분명", "사업장명", "의료인수", "경도", "위도", "도로명우편번호", "소재지전체주소", "도로명전체주소", "인허가일자", "폐업일자", "영업상태명"
]]

# 5️⃣ 새로운 엑셀 파일 저장
output_path = "Bogun.xlsx"
filtered_df.to_excel(output_path, index=False)

print(f"✅ 변환된 엑셀 파일이 저장되었습니다: {output_path}")

from tqdm import tqdm

# 네이버 API 키 설정
CLIENT_ID = "qwt3tw05k9"
CLIENT_SECRET = "GBfbsLbGQIte7gkUgxW5QKvSnE62EfNuouwtoPJq"
REVERSE_GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"

# 엑셀 파일 로드
Bogun_file_path = "Bogun.xlsx"
df = pd.read_excel(Bogun_file_path)

# 네이버 API를 이용해 시도 및 시군구 정보를 가져오는 함수
def get_address_from_naver(lat, lon):
    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
        "X-NCP-APIGW-API-KEY": CLIENT_SECRET
    }
    params = {
        "coords": f"{lon},{lat}",
        "output": "json",
        "orders": "legalcode"
    }
    response = requests.get(REVERSE_GEOCODE_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        try:
            region = data["results"][0]["region"]
            sido = region["area1"]["name"]
            sigungu = region["area2"]["name"]
            return sido, sigungu
        except (IndexError, KeyError):
            return None, None
    return None, None

# 새로운 데이터프레임 생성
new_data = []
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='네이버 API 요청 진행 중'):
    sido, sigungu = get_address_from_naver(row["위도"], row["경도"])
    new_data.append([row["사업장명"], row["업태구분명"], row["의료인수"], row["위도"], row["경도"], sido, sigungu, row["도로명전체주소"], row["소재지전체주소"], row["도로명우편번호"]])

new_df = pd.DataFrame(new_data, columns=["사업장명", "분류", "의료인수", "위도", "경도", "시도", "시군구", "도로명전체주소", "소재지전체주소", "도로명우편번호"])

# 새로운 엑셀 파일 저장
output_path = "processed_Bogun.xlsx"
new_df.to_excel(output_path, index=False)
print(f"파일 저장 완료: {output_path}")

import requests
import pandas as pd
import pyproj
from tqdm import tqdm
import time

# 네이버 API 키 설정
CLIENT_ID = "qwt3tw05k9"
CLIENT_SECRET = "GBfbsLbGQIte7gkUgxW5QKvSnE62EfNuouwtoPJq"
GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
REVERSE_GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"

# 엑셀 파일 로드
file_path = "processed_Bogun.xlsx"
df = pd.read_excel(file_path)

# 네이버 API를 이용해 주소를 위경도로 변환하는 함수
def get_lat_lon_from_naver(address):
    headers = {
        "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
        "X-NCP-APIGW-API-KEY": CLIENT_SECRET
    }
    params = {"query": address}
    response = requests.get(GEOCODE_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if "addresses" in data and len(data["addresses"]) > 0:
            return float(data["addresses"][0]["y"]), float(data["addresses"][0]["x"])
    return None, None

# 위경도 데이터가 없는 경우 주소를 이용해 채우기
tqdm.pandas(desc="주소를 위경도로 변환 중")
for index, row in df.iterrows():
    if pd.isna(row["위도"]) or pd.isna(row["경도"]):
        address = row["도로명전체주소"] if pd.notna(row["도로명전체주소"]) else row["소재지전체주소"]
        if pd.notna(address):
            lat, lon = get_lat_lon_from_naver(address)
            df.at[index, "위도"] = lat
            df.at[index, "경도"] = lon
            time.sleep(0.1)  # API 요청 속도 제한 방지

# 시도 및 시군구가 없는 경우 위경도를 이용해 채우기
tqdm.pandas(desc="위경도로 시도 및 시군구 변환 중")
for index, row in df.iterrows():
    if pd.isna(row["시도"]) or pd.isna(row["시군구"]):
        if pd.notna(row["위도"]) and pd.notna(row["경도"]):
            sido, sigungu = get_address_from_naver(row["위도"], row["경도"])
            df.at[index, "시도"] = sido
            df.at[index, "시군구"] = sigungu
            time.sleep(0.1)  # API 요청 속도 제한 방지

# 업데이트된 데이터 저장
output_path = "processed_Bogun_updated.xlsx"
df.to_excel(output_path, index=False)
print(f"업데이트된 파일 저장 완료: {output_path}")

# 📌 파일 로드
file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(file_path)

Bogun_file_path = "processed_Bogun_updated.xlsx"
df_Bogun = pd.read_excel(Bogun_file_path)

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
df_Bogun["가중치_의료인수"] = df_Bogun["의료인수"]

# 📌 시군구 통합 컬럼 생성
df_hanmed["시군구_통합"] = (df_hanmed["시도"] + " " + df_hanmed["시군구"]).str.strip()
df_clinic["시군구_통합"] = (df_clinic["시도"] + " " + df_clinic["시군구"]).str.strip()
df_Bogun["시군구_통합"] = (df_Bogun["시도"] + " " + df_Bogun["시군구"]).str.strip()

# 📌 한의원 및 의원 의료인수 총합 계산
hanmed_ratio = df_hanmed.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
clinic_ratio = df_clinic.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
Bogun_ratio = df_Bogun.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()


# 📌 컬럼명 변경
hanmed_ratio.rename(columns={"가중치_의료인수": "한의원 의료인수 총합"}, inplace=True)
clinic_ratio.rename(columns={"가중치_의료인수": "의원 의료인수 총합"}, inplace=True)
Bogun_ratio.rename(columns={"가중치_의료인수": "보건원 의료인수 총합"}, inplace=True)


# 📌 보건원 데이터 병합
df_med_ratio = pd.merge(Bogun_ratio, hanmed_ratio, on="시군구_통합", how="outer").fillna(0)

# 📌 기본 데이터 병합
df_med_ratio = pd.merge(df_med_ratio, clinic_ratio, on="시군구_통합", how="outer").fillna(0)


# 📌 전체 의료기관 의료인수 계산 (한의원 + 의원)
df_med_ratio["전체 의료기관 의료인수"] = df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"]

# 📌 한의원 비율
df_med_ratio["한의원 비율"] = df_med_ratio["한의원 의료인수 총합"] / df_med_ratio["전체 의료기관 의료인수"]

# ✅ '노인 비율' 데이터 병합
df_med_ratio = pd.merge(df_med_ratio, df_age[["시군구_통합", "노인 비율", "총 인구수"]], on="시군구_통합", how="left")

# 📌 보건원 비율
df_med_ratio["인구당 보건원 의료인수"] = df_med_ratio["보건원 의료인수 총합"] / df_med_ratio["총 인구수"]


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

# ✅ 최종적으로 정확히 일치한 의원 개수 확인
print(f"✅ 정확히 일치하는 의원 개수: {df_chronic_clinic.shape[0]}")

print("🔍 df_chronic_clinic 컬럼 목록:", df_chronic_clinic.columns.tolist())


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
for col in ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율", "인구당 보건원 의료인수"]:
    sns.regplot(x=df_med_ratio["노인 비율"], y=df_med_ratio[col], label=col, scatter_kws={'alpha': 0.5})

print("🔍 df_med_ratio 컬럼 목록:", df_med_ratio.columns)
print(df_med_ratio.head())  # 상위 5개 행 출력하여 값 확인

# ✅ 피어슨 상관 분석
correlation_results = [(col, *pearsonr(df_med_ratio["노인 비율"], df_med_ratio[col])) for col in
                       ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율", "인구당 보건원 의료인수"]]
df_correlation = pd.DataFrame(correlation_results, columns=["변수", "피어슨 상관계수 (r)", "p-value"])

# ✅ 각 의료기관 유형별 개수(n) 및 총 의료인 수(nn) 계산
num_hanmed = df_hanmed.shape[0]  # 한의원 개수
num_Bogun = df_Bogun.shape[0]
num_internal_medicine = df_internal_medicine.shape[0]  # 내과의원 개수
num_family_medicine = df_family_medicine.shape[0]  # 가정의학과의원 개수
num_non_specialized = df_non_specialized.shape[0]  # 미표방 의원 개수
num_chronic_care = df_chronic_clinic.shape[0]  # 시범사업 참여의원 개수

# ✅ 총 의료인 수 계산
sum_hanmed = df_hanmed["의료인수"].sum()  # 한의원 총 의료인수
sum_Bogun = df_Bogun["의료인수"].sum()
sum_internal_medicine = df_internal_medicine["의료인수"].sum()  # 내과의원 총 의료인수
sum_family_medicine = df_family_medicine["의료인수"].sum()  # 가정의학과의원 총 의료인수
sum_non_specialized = df_non_specialized["의료인수"].sum()  # 미표방 의원 총 의료인수
sum_chronic_care = df_chronic_clinic["의료인수"].sum()  # 시범사업 참여의원 총 의료인수

# ✅ 개수(n) 및 총 의료인수(nn) 리스트 생성
num_values = [num_hanmed, num_internal_medicine, num_family_medicine, num_non_specialized, num_chronic_care, num_Bogun]
num_medical_staff_values = [sum_hanmed, sum_internal_medicine, sum_family_medicine, sum_non_specialized, sum_chronic_care, sum_Bogun]

# ✅ df_correlation에 개수(n) 및 의료인수(nn) 추가
df_correlation["의료기관 개수 (n)"] = num_values
df_correlation["총 의료인 수 (nn)"] = num_medical_staff_values

# ✅ 최종 결과 출력
print(df_correlation)

import matplotlib.pyplot as plt
import seaborn as sns

# ✅ 메인 그래프 설정 (왼쪽 Y축)
fig, ax1 = plt.subplots(figsize=(10, 6))

colors = ['b', 'g', 'r', 'c', 'm']  # 그래프 색상 지정
columns = ["한의원 비율", "내과의원 비율", "가정의학과의원 비율", "미표방 의원 비율", "시범사업 참여의원 비율"]

# ✅ 비율 데이터를 왼쪽 Y축에 플로팅
for i, col in enumerate(columns):
    sns.regplot(x=df_med_ratio["노인 비율"], y=df_med_ratio[col], ax=ax1, label=col, scatter=False, color=colors[i])

ax1.set_xlabel("노인 비율")
ax1.set_ylabel("의료기관 비율 (0~1)")
ax1.legend(loc="upper left")

# ✅ 보조 Y축 생성 (오른쪽 Y축)
ax2 = ax1.twinx()
sns.regplot(x=df_med_ratio["노인 비율"], y=df_med_ratio["인구당 보건원 의료인수"], ax=ax2, scatter=False, color='orange')

ax2.set_ylabel("인구당 보건원 의료인수 (0~1)")
ax2.legend(loc="upper right")

# ✅ 그래프 제목 설정 및 저장
plt.title("노인 비율과 의료기관 유형 간 관계 (이중 Y축)")
plt.savefig("이중Y축_그래프.png", dpi=300, bbox_inches='tight')
plt.show()


df_correlation_sorted = df_correlation.sort_values(by="피어슨 상관계수 (r)", ascending=False)

# ✅ 피어슨 상관계수 막대그래프 (큰 값부터 정렬)
plt.figure(figsize=(8, 5))
sns.barplot(
    x="변수", y="피어슨 상관계수 (r)", data=df_correlation_sorted, hue="변수", palette="coolwarm", legend=False
)

# ✅ 그래프 꾸미기
plt.axhline(0, color="black", linewidth=1)  # 0 기준선 추가
plt.xticks(rotation=45)  # X축 라벨 기울여서 가독성 향상
plt.ylabel("피어슨 상관계수 (r)")
plt.title("노인 비율과 의료기관 유형 간 피어슨 상관계수 비교 (내림차순 정렬)")

# ✅ 그래프 저장
plt.savefig("피어슨_막대그래프_정렬.png", dpi=300, bbox_inches='tight')

# ✅ 그래프 표시
plt.show()