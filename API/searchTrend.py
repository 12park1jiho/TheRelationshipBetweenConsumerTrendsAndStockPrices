import urllib.request
import json
import pandas as pd

# API 정보
client_id = "UxnarqmjJkR4PJ4TkID1"  # 네이버 API 클라이언트 ID
client_secret = "DBU9EjqyzR"  # 네이버 API 클라이언트 시크릿
url = "https://openapi.naver.com/v1/search/news.json"  # JSON 형식으로 요청

# 검색어 및 파라미터
query = "2023 1월 1일 삼성전자 뉴스"  # 검색어
display = 100  # 검색 결과 개수
start = 1  # 검색 시작 위치
sort = "date"  # 유사도순 정렬 (기본값)

# URL 생성
query = urllib.parse.quote(query)  # 검색어 URL 인코딩
params = f"?query={query}&display={display}&start={start}&sort={sort}"
full_url = url + params

# 요청 설정
request = urllib.request.Request(full_url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)

try:
    response = urllib.request.urlopen(request)
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()
        data = json.loads(response_body.decode('utf-8'))

        # 데이터 활용 (예시: pandas DataFrame으로 변환 후 CSV 파일로 저장)
        items = data['items']
        df = pd.DataFrame(items)

        # 필요에 따라 데이터 정제 및 추가 작업 수행
        df = df[['title', 'link', 'description', 'pubDate']]  # 예시: 필요한 열만 선택

        df.to_csv("naver_news_data.csv", encoding="utf-8-sig", index=False)  # CSV 파일로 저장
        print("데이터가 naver_news_data.csv 파일에 저장되었습니다.")

    else:
        print(f"Error Code: {rescode}")

except Exception as e:
    print(f"Error: {e}")