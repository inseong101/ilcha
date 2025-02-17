import pandas as pd

# 파일 로드
file_path_hospitals = "processed_hospitals_updated.xlsx"
file_path_chronic_care = "국민건강보험공단_일차의료 만성질환관리 시범사업 참여의원 목록_20240331.csv"

# 파일 불러오기
df_hospitals = pd.read_excel(file_path_hospitals)
df_chronic_care = pd.read_csv(file_path_chronic_care, encoding="cp949")

# 🚨 1️⃣ 공백 제거 후 비교 (완전 일치 비교)
df_hospitals["사업장명"] = df_hospitals["사업장명"].str.strip()
df_chronic_care["요양기관명"] = df_chronic_care["요양기관명"].str.strip()

# 🚨 2️⃣ 각 요양기관명에 대해 사업장명에서 일치하는 경우 찾기
matched_indices = df_chronic_care["요양기관명"].apply(lambda x: x in df_hospitals["사업장명"].values)
matched_df = df_chronic_care[matched_indices]  # 일치하는 기관만 선택

# 🚨 3️⃣ 국민건강보험공단 파일에는 있지만 processed_hospitals에 없는 요양기관명 찾기
missing_hospitals = set(df_chronic_care["요양기관명"]) - set(df_hospitals["사업장명"])

# ✅ 결과 출력
print(f"✅ 국민건강보험공단 파일의 전체 요양기관 수: {len(df_chronic_care)}")  # 2609개 예상
print(f"✅ 정확히 일치하는 의원 수: {len(matched_df)}")  # 2609 이하여야 정상
print(f"✅ processed_hospitals에 없는 요양기관명 수: {len(missing_hospitals)}")

# ✅ 없는 요양기관명 개별 출력
print("\n❌ processed_hospitals에 없는 요양기관명 목록:")
for hospital in sorted(missing_hospitals):  # 가나다순 정렬 후 출력
    print(hospital)

# ✅ 파일로 저장
matched_df.to_excel("정확히_일치하는_의원목록.xlsx", index=False)
pd.DataFrame({"없는 요양기관명": list(missing_hospitals)}).to_excel("없는_의원목록.xlsx", index=False)

print("\n✅ 분석 완료! '정확히_일치하는_의원목록.xlsx', '없는_의원목록.xlsx' 파일을 확인하세요.")