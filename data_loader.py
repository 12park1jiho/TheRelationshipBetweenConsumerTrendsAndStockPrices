# data_loader.py
import pandas as pd
import os

def load_company_names(csv_path="data/crno.csv"):
    """CSV 파일에서 회사 이름 목록을 로드합니다."""
    try:
        df_csv = pd.read_csv(csv_path)
        stock_names = df_csv["회사명"].tolist()
        return stock_names
    except FileNotFoundError:
        print(f"CSV 파일({csv_path})을 찾을 수 없습니다.")
        return None
    except KeyError:
        print(f"CSV 파일에 '회사명' 열이 없습니다.")
        return None

def create_directories():
    """데이터 및 결과 저장 폴더를 생성합니다."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("results"):
        os.makedirs("results")

def load_stock_data(stock_names):
    """회사별 주가 데이터를 로드하고 딕셔너리에 저장합니다."""
    stock_data_dict = {}
    df_stock_monthly_dict = {}
    for stock_name in stock_names:
        stock_file_name = f"data/stock_data_{stock_name}_2019_2023.csv"
        try:
            stock_data = pd.read_csv(stock_file_name)
            stock_data_dict[stock_name] = stock_data
            df_stock_monthly = stock_data.groupby('basDt').mean(numeric_only=True).reset_index() # 월별 평균 주가 계산
            df_stock_monthly_dict[stock_name] = df_stock_monthly # 딕셔너리에 저장
        except FileNotFoundError:
            stock_data_dict[stock_name] = None
            df_stock_monthly_dict[stock_name] = None
            print(f"  주가 데이터 '{stock_file_name}' 파일을 찾을 수 없습니다.") # 로그 메시지 추가
    return stock_data_dict, df_stock_monthly_dict

def load_financial_data(stock_names):
    """회사별 재무 데이터를 로드하고 딕셔너리에 저장합니다."""
    financial_data_dict = {}
    for stock_name in stock_names:
        financial_file_name = f"data/financial_data_{stock_name}_2019_2023.csv"
        try:
            with open(financial_file_name, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                financial_columns = header_line.split(',')
                columns_to_use = [
                    col for col in financial_columns
                    if col not in ['fnclDcdNm', 'curCd']
                ]
            financial_data = pd.read_csv(financial_file_name, usecols=columns_to_use)
            financial_data_dict[stock_name] = financial_data
            print(f"  재무 데이터 '{financial_file_name}' 로드 완료 (fnclDcdNm, curCd 컬럼 제외)") # 로그 메시지 추가
        except FileNotFoundError:
            financial_data_dict[stock_name] = None
            print(f"  재무 데이터 '{financial_file_name}' 파일을 찾을 수 없습니다.") # 로그 메시지 추가
    return financial_data_dict

def load_consumer_data(csv_path="data/소비자동향조사(전국, 월, 2008.9~)_05204658.csv"):
    """소비자 동향 조사 데이터를 로드합니다."""
    try:
        consumer_data = pd.read_csv(csv_path)
        return consumer_data
    except FileNotFoundError:
        print(f"소비자동향조사 CSV 파일({csv_path})을 찾을 수 없습니다.")
        return None

def convert_date_format(consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict, stock_names):
    """날짜 컬럼의 형식을 datetime으로 변환합니다."""
    if consumer_data is not None:
        consumer_data['basDt'] = pd.to_datetime(consumer_data['날짜'], format='%Y%m')
        consumer_data['basDt'] = consumer_data['basDt'].dt.to_period('M')
    for stock_name in stock_names:
        if stock_data_dict[stock_name] is not None:
            stock_data_dict[stock_name]['basDt'] = pd.to_datetime(stock_data_dict[stock_name]['basDt'], format='%Y%m%d')
            stock_data_dict[stock_name]['basDt'] = stock_data_dict[stock_name]['basDt'].dt.to_period('M')
        if financial_data_dict[stock_name] is not None:
            financial_data_dict[stock_name]['basDt'] = pd.to_datetime(
                financial_data_dict[stock_name]['basDt'].astype(str), format='%Y%m%d')
            financial_data_dict[stock_name]['basDt'] = financial_data_dict[stock_name]['basDt'].dropna().dt.to_period('M')  # NaN 제거 후 Period 변환
        if df_stock_monthly_dict[stock_name] is not None:
            df_stock_monthly_dict[stock_name]['basDt'] = pd.to_datetime(df_stock_monthly_dict[stock_name]['basDt'],format='%Y%m%d')  # 월별 평균 주가 basDt format 변경
            df_stock_monthly_dict[stock_name]['basDt'] = df_stock_monthly_dict[stock_name]['basDt'].dt.to_period('M')  # 수정: 불필요한 ['basDt'] 인덱싱 제거

    return consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict

# (main.py에서 호출 예시)
# stock_names = load_company_names()
# create_directories()
# stock_data_dict, df_stock_monthly_dict = load_stock_data(stock_names)
# financial_data_dict = load_financial_data(stock_names)
# consumer_data = load_consumer_data()
# consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict = convert_date_format(consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict, stock_names)