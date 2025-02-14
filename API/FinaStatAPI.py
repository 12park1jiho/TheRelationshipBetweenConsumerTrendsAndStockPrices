import requests
import xmltodict
import pandas as pd
import csv
import os

# API 서비스 정보
url = 'http://apis.data.go.kr/1160100/service/GetFinaStatInfoService_V2/getSummFinaStat_V2'
api_key = 'GiG8n+kvj4BXhqhM4pn6NhAxeszpUPoamserVlEFbVZ/0N4Ldkjh0QWCYIJlOEEHJJx7tJverG3L+58qsIZfRQ=='  # API 키

# CSV 파일 읽어오기
try:
    df_csv = pd.read_csv("../data/crno.csv")  # CSV 파일 경로 수정
    crno_list = df_csv["법인등록번호"].tolist()  # "법인등록번호" 열의 값들을 리스트로 추출
    company_names = df_csv["회사명"].tolist()  # "회사명" 열의 값들을 리스트로 추출
except FileNotFoundError:
    print("CSV 파일(data/crno.csv)을 찾을 수 없습니다.")
    exit()
except KeyError:
    print("CSV 파일에 '법인등록번호' 또는 '회사명' 열이 없습니다.")
    exit()

# "data" 폴더 생성 (존재하지 않을 경우)
if not os.path.exists("../data"):
    os.makedirs("../data")

# 전체 데이터를 저장할 리스트
all_data = []

# 회사명과 법인등록번호 매핑 딕셔너리 생성
company_crno_map = dict(zip(company_names, crno_list))

for company_name, crno in company_crno_map.items():
    for year in range(2019, 2024):
        params = {
            'serviceKey': api_key,
            'numOfRows': 100,  # 한 페이지 결과 수
            'pageNo': 1,  # 페이지 번호
            'resultType': 'json',  # 결과 형식
            'bizYear': year,  # 사업연도
            'crno': crno,  # 법인등록번호
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

            data = response.json()

            if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
                items = data['response']['body']['items']['item']
                if isinstance(items, list):
                    all_data.extend(items)
                else:
                    all_data.append(items)
            else:
                print(f"{company_name} {year}년 데이터 없음 또는 응답 구조 변경됨")

        except requests.exceptions.RequestException as e:
            print(f"{company_name} {year}년 API 호출 에러: {e}")
        except (KeyError, TypeError) as e:
            print(f"{company_name} {year}년 데이터 처리 에러: {e}")

    # 데이터프레임으로 변환 및 CSV 파일 저장 (회사별로 저장)
    if all_data:
        df_result = pd.DataFrame(all_data)
        file_name = f"../data/financial_data_{company_name}_{2019}_{2023}.csv"  # 파일 이름 형식 변경
        df_result.to_csv(file_name, index=False, encoding="utf-8")  # UTF-8 인코딩 추가
        print(f"{company_name} 데이터가 {file_name} 파일에 저장되었습니다.")
        all_data = []  # 다음 회사를 위해 데이터 초기화
    else:
        print(f"{company_name} 가져온 데이터가 없습니다.")