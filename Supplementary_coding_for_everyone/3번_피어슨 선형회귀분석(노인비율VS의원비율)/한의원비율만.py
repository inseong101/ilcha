import pyproj
import pandas as pd
import matplotlib.pyplot as plt

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

# ✅ 한글 폰트 설정 (Windows: 'Malgun Gothic', Mac: 'AppleGothic')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# 1️⃣ 엑셀 파일 로드
file_path = "지방행정인허가에서_다운받은_파일.xlsx"
#다운받는 곳: https://www.localdata.go.kr/devcenter/dataDown.do?menuNo=20001 여기서 맨 밑에 의원 xlsx로 다운받으세요.
df = pd.read_excel(file_path, engine="openpyxl")

# 2️⃣ "의원" 또는 "한의원"에 해당하고, "영업상태명"이 "영업/정상"인 데이터 필터링
filtered_df = df[
    (df["의료기관종별명"].isin(["의원", "한의원"])) & (df["영업상태명"] == "영업/정상")
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
output_path = "filtered_hospitals_converted.xlsx"
filtered_df.to_excel(output_path, index=False)

print(f"✅ 변환된 엑셀 파일이 저장되었습니다: {output_path}")

#수고하셨습니다. 지방행정인허가에서 영업중인 의원과 한의원의 의료인수와 위치를 알아냈습니다.

from tqdm import tqdm

# 네이버 API 키 설정
CLIENT_ID = "qwt3tw05k9"
CLIENT_SECRET = "GBfbsLbGQIte7gkUgxW5QKvSnE62EfNuouwtoPJq"
REVERSE_GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc"

# 엑셀 파일 로드
file_path = "filtered_hospitals_converted.xlsx"
df = pd.read_excel(file_path)

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
output_path = "processed_hospitals.xlsx"
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
file_path = "processed_hospitals.xlsx"
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
output_path = "processed_hospitals_updated.xlsx"
df.to_excel(output_path, index=False)
print(f"업데이트된 파일 저장 완료: {output_path}")

#수고하셨습니다. NAVER MAPS API를 이용해서 위경도 데이터로 시군구 데이터를 알아냈습니다.

### 1️⃣ 한의원 비율 계산 ###
medical_file_path = "processed_hospitals_updated.xlsx"
df_medical = pd.read_excel(medical_file_path)

# 🔹 한의원 및 의원 데이터 필터링
df_hanmed = df_medical[df_medical["분류"] == "한의원"].copy()
df_clinic = df_medical[df_medical["분류"] == "의원"].copy()

# 🔹 의료인수 가중치 적용
df_hanmed["가중치_의료인수"] = df_hanmed["의료인수"]
df_clinic["가중치_의료인수"] = df_clinic["의료인수"]

# 🔹 "시도 + 시군구" 결합 후 공백 제거
df_hanmed["시군구"] = df_hanmed["시군구"].fillna("")
df_clinic["시군구"] = df_clinic["시군구"].fillna("")

df_hanmed["시군구_통합"] = (df_hanmed["시도"] + " " + df_hanmed["시군구"]).str.strip()
df_clinic["시군구_통합"] = (df_clinic["시도"] + " " + df_clinic["시군구"]).str.strip()

# 🔹 각 시군구별 한의원 및 의원 의료인수 총합 계산
hanmed_ratio = df_hanmed.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()
clinic_ratio = df_clinic.groupby("시군구_통합")["가중치_의료인수"].sum().reset_index()

# 🔹 컬럼명 변경
hanmed_ratio.rename(columns={"가중치_의료인수": "한의원 의료인수 총합"}, inplace=True)
clinic_ratio.rename(columns={"가중치_의료인수": "의원 의료인수 총합"}, inplace=True)

# 🔹 한의원 + 의원 데이터 병합
df_med_ratio = pd.merge(hanmed_ratio, clinic_ratio, on="시군구_통합", how="outer").fillna(0)

# 🔹 한의원 비율 계산 (전체 의료기관 대비 한의원 비율)
df_med_ratio["한의원 비율"] = df_med_ratio["한의원 의료인수 총합"] / (
    df_med_ratio["한의원 의료인수 총합"] + df_med_ratio["의원 의료인수 총합"]
)

import pandas as pd
from tqdm import tqdm


def preprocess_population_data(file_path, output_path):
    # 엑셀 파일 불러오기
    df = pd.read_excel(file_path, dtype=str)  # 모든 데이터를 문자열로 로드하여 처리

    # 첫 3행 삭제
    df = df.iloc[3:].reset_index(drop=True)

    # 컬럼명 변경
    df.columns = ["행정기관코드", "행정기관", "총 인구수", "연령구간인구수", "0~9세", "10~19세", "20~29세",
                  "30~39세", "40~49세", "50~59세", "60~69세", "70~79세", "80~89세", "90~99세", "100세 이상"]

    # '행정기관코드' 칼럼 삭제
    df = df.drop(columns=["행정기관코드", "연령구간인구수"])

    # 공백 제거 (맨 뒤 공백 포함)
    df["행정기관"] = df["행정기관"].str.strip()

    # '출장소'가 포함된 행 삭제
    removed_office_rows = df[df["행정기관"].str.contains("출장소")]["행정기관"].tolist()
    df = df[~df["행정기관"].str.contains("출장소")]
    print("삭제된 출장소 행:", removed_office_rows)

    # 공백 제거 후 단어 하나짜리 행 삭제 (세종특별자치시는 예외)
    single_word_rows = df[(df["행정기관"].str.count(" ") == 0) & (df["행정기관"] != "세종특별자치시")]
    removed_single_word_rows = single_word_rows["행정기관"].tolist()
    df = df.drop(single_word_rows.index)
    print("삭제된 단어 하나짜리 행:", removed_single_word_rows)

    # '시' 단위 데이터 삭제 (해당 시에 속하는 '시군구'가 존재하는 경우 '시' 삭제)
    main_cities_to_delete = [
        '경상북도 포항시', '경기도 수원시', '경기도 안양시', '경기도 성남시', '경기도 안산시',
        '충청북도 청주시', '경기도 고양시', '경상남도 창원시', '충청남도 천안시', '경기도 용인시',
        '경기도 부천시', '전북특별자치도 전주시'
    ]

    to_delete = []
    for city in main_cities_to_delete:
        sub_regions = df[df["행정기관"].str.startswith(city + " ")]["행정기관"].tolist()
        if sub_regions:  # '시군구' 데이터가 있는 경우에만 삭제
            to_delete.append(city)

    df = df[~df["행정기관"].isin(to_delete)]

    if to_delete:
        print("삭제된 '시' 단위 행 (시군구가 존재하는 경우 삭제):", to_delete)

    # 숫자 변환 (쉼표 제거 후 NaN 처리 및 변환 - float 유지)
    numeric_cols = ["총 인구수", "60~69세", "70~79세", "80~89세", "90~99세", "100세 이상"]
    for col in tqdm(numeric_cols, desc='숫자 데이터 변환 진행 중'):
        df[col] = df[col].str.replace(",", "", regex=True).astype(float).fillna(0)

    # 노인 비율 계산 (0~1 사이 값, 유효숫자 4개 유지)
    df["노인 비율"] = ((df["60~69세"] + df["70~79세"] + df["80~89세"] + df["90~99세"] + df["100세 이상"])
                   / df["총 인구수"]).round(4)

    # 필요한 칼럼만 유지
    df = df[["행정기관", "노인 비율", "총 인구수"]]

    # 결과 저장 (xlsxwriter가 없을 경우 openpyxl 사용)
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
            worksheet = writer.sheets['Sheet1']
            for col_num, value in tqdm(enumerate(df.columns.values), total=len(df.columns), desc='엑셀 저장 진행 중'):
                worksheet.write(0, col_num, value)
    except ModuleNotFoundError:
        df.to_excel(output_path, index=False, engine='openpyxl')

    print("전처리된 데이터가 저장되었습니다:", output_path)

# 사용 예시
input_file = "행안부에서_다운받은_파일.xlsx"
output_file = "전처리된_노인비율.xlsx"
preprocess_population_data(input_file, output_file)

age_population_file = "전처리된_노인비율.xlsx"
df_age = pd.read_excel(age_population_file)

# 🔹 필요한 컬럼만 선택 (행정기관, 노인 비율)
df_age = df_age[["행정기관", "노인 비율"]].copy()

# 🔹 행정기관 공백 제거 (시군구 통합 시 공백 차이 문제 해결)
df_age["시군구_통합"] = df_age["행정기관"].str.strip()

# 🔹 데이터 형식 변환 (쉼표 제거)
df_age["노인 비율"] = df_age["노인 비율"].astype(str).str.replace(",", "").astype(float)

### 3️⃣ 데이터 병합 ###
# 🔹 "시군구_통합" 컬럼을 기준으로 병합
df_merged = pd.merge(df_med_ratio, df_age, on="시군구_통합", how="inner")

# 🔹 병합되지 않은 지역 확인
missing_in_hanmed = set(df_age["시군구_통합"].unique()) - set(df_med_ratio["시군구_통합"].unique())
missing_in_age = set(df_med_ratio["시군구_통합"].unique()) - set(df_age["시군구_통합"].unique())

print("한의원 데이터에는 있지만, 노인 비율 데이터에는 없는 시군구:", missing_in_hanmed)
print("노인 비율 데이터에는 있지만, 한의원 데이터에는 없는 시군구:", missing_in_age)

# 🔹 병합 후 데이터 개수 확인
print(f"한의원 및 의원 데이터 개수: {len(df_med_ratio)}")
print(f"노인 비율 데이터 개수: {len(df_age)}")
print(f"병합된 데이터 개수: {len(df_merged)}")

# 🔹 NaN 값 제거
df_merged = df_merged.dropna(subset=["한의원 비율", "노인 비율"])

# 5️⃣ 데이터 저장 ###
df_merged.to_excel("한의원_비율_노인 비율_상관분석.xlsx", index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ 폰트 설정
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# ✅ 그래프 그리기
plt.figure(figsize=(8, 6))

# ✅ Scatter plot 스타일 수정 (투명한 원)
sns.scatterplot(x=df_merged["노인 비율"], y=df_merged["한의원 비율"],
                edgecolor='blue', facecolor='none', alpha=0.7)

# ✅ 선형 회귀선 (신뢰구간 제거)
sns.regplot(x=df_merged["노인 비율"], y=df_merged["한의원 비율"],
            scatter=False, ci=None, color="red", line_kws={"linewidth": 2})

# ✅ 축 설정 (폰트 크기 조정)
plt.xlabel("Proportion of elderly population", fontsize=14)
plt.ylabel("Proportion of Korean medicine clinics", fontsize=14)

# ✅ 그리드 추가
plt.grid(True, linestyle="--", alpha=0.5)

# ✅ 그래프 저장 (500 dpi 고해상도)
save_path = "/Supplementary_coding_for_everyone/3번_피어슨 선형회귀분석(노인비율VS의원비율)"
save_file = os.path.join(save_path, "Elderly_vs_KoreanMedicine.png")
plt.savefig(save_file, dpi=500, bbox_inches='tight')
plt.show()

print(f"✅ 그래프 저장 완료: {save_file}")