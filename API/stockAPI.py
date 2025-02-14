import requests
import xmltodict
import pandas as pd
from urllib.parse import urlencode
from time import sleep
import csv
import os

# API 서비스 정보
url = 'http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo'
api_key = 'GiG8n+kvj4BXhqhM4pn6NhAxeszpUPoamserVlEFbVZ/0N4Ldkjh0QWCYIJlOEEHJJx7tJverG3L+58qsIZfRQ=='  # 본인의 API 키

# CSV 파일 읽어오기
try:
    df_csv = pd.read_csv("../data/crno.csv")  # CSV 파일 경로 수정
    stock_names = df_csv["회사명"].tolist()  # "회사명" 열의 값들을 리스트로 추출
except FileNotFoundError:
    print("CSV 파일(../data/crno.csv)을 찾을 수 없습니다.")
    exit()
except KeyError:
    print("CSV 파일에 '회사명' 열이 없습니다.")
    exit()

# "data" 폴더 생성 (존재하지 않을 경우)
if not os.path.exists("../data"):
    os.makedirs("../data")

# 데이터 수집 및 저장
for stock_name in stock_names:
    all_data = []
    page_no = 1
    num_of_rows = 100  # API가 한 번에 가져올 수 있는 최대 개수

    while True:
        params = {
            'serviceKey': api_key,
            'numOfRows': num_of_rows,
            'pageNo': page_no,
            'resultType': 'json',
            'beginBasDt': '20190101',
            'endBasDt': '20231231',
            'itmsNm': stock_name,
        }

        response = requests.get(f"{url}?{urlencode(params)}")
        response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
        data = response.json()

        # API 응답이 정상인지 확인
        if data['response']['header']['resultCode'] != '00':
            print(f"{stock_name} API 호출 실패: {data['response']['header']['resultMsg']}")
            break

        # 데이터 추출
        items = data['response']['body'].get('items', {}).get('item', [])
        if not items:
            print(f"{stock_name} 더 이상 가져올 데이터가 없습니다.")
            break

        all_data.extend(items)
        print(f"{stock_name} {len(items)}개 데이터 수집 완료 (페이지 {page_no})")

        # 다음 페이지로 이동
        page_no += 1
        sleep(1)  # API 요청 제한 방지를 위해 1초 대기

    # 데이터프레임 변환 및 저장
    if all_data:
        df = pd.DataFrame(all_data)
        file_name = f'../data/stock_data_{stock_name}_{2019}_{2023}.csv'  # "data" 폴더에 저장
        df.to_csv(file_name, index=False, encoding="utf-8")  # 'index=False'로 인덱스를 저장하지 않음, UTF-8 인코딩 추가
        print(f"{stock_name} 데이터가 {file_name} 파일에 저장되었습니다.")
    else:
        print(f"{stock_name} 수집된 데이터가 없습니다.")