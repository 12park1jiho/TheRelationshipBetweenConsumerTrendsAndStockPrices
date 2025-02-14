# main.py
import pandas as pd
from data_loader import load_company_names, create_directories, load_stock_data, load_financial_data, load_consumer_data, convert_date_format
from feature_engineering import create_lagged_feature, create_rolling_feature, create_financial_ratio_feature
from visualization import plot_model_performance_bar_chart
from stock_price_analysis import analyze_stock_price
from financial_analysis import analyze_financial_data
import os

# --- 1. 데이터 로드 및 준비 ---
stock_names = load_company_names() # 회사 이름 목록 로드
if stock_names is None: # 회사 이름 목록 로드 실패 시
    exit()

create_directories() # 데이터/결과 저장 폴더 생성
stock_data_dict, df_stock_monthly_dict = load_stock_data(stock_names) # 주가 데이터 로드
financial_data_dict = load_financial_data(stock_names) # 재무 데이터 로드
consumer_data = load_consumer_data() # 소비자 동향 조사 데이터 로드
consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict = convert_date_format(consumer_data, stock_data_dict, financial_data_dict, df_stock_monthly_dict, stock_names) # 날짜 format 변환


# 회사별 결과 저장을 위한 딕셔너리 (전체 회사 결과 통합)
sp_results = {} # 주가 예측 결과 저장
fi_results = {} # 재무 예측 결과 저장


# --- 2. 회사별 데이터 분석 ---
for stock_name in stock_names:
    print(f"----- {stock_name} 데이터 로드 및 분석 시작 -----")

    # 데이터 로드 (data_loader 모듈에서 딕셔너리 형태로 로드 완료)
    stock_data = stock_data_dict[stock_name]
    financial_data = financial_data_dict[stock_name]
    df_stock_monthly = df_stock_monthly_dict[stock_name]


    # --- Feature Engineering ---
    print("\n  --- Feature Engineering 시작 ---")

    if df_stock_monthly is not None: # 주가 Lagged/Rolling Feature 생성
        df_stock_monthly = create_lagged_feature(df_stock_monthly, 'clpr', lags=[1, 3])
        df_stock_monthly = create_rolling_feature(df_stock_monthly, 'clpr', window=3, stat_funcs=['mean', 'std'])
        df_stock_monthly = create_rolling_feature(df_stock_monthly, 'clpr', window=7, stat_funcs=['mean', 'std'])
        print("    주가 Feature Engineering 완료")
    if consumer_data is not None: # 소비자동향 Lagged Feature 생성
        consumer_data = create_lagged_feature(consumer_data, '소비자심리지수', lags=[1, 3])
        print("    소비자동향 Lagged Feature 생성 완료")
    if financial_data is not None: # 재무 비율 Feature 생성
        financial_data = create_financial_ratio_feature(financial_data)
        print("    재무 비율 Feature 생성 완료")

    print("  --- Feature Engineering 완료 ---")


    # --- [분석 1] 소비자동향 + 주가 데이터 분석 ---
    print(f"\n  --- [분석 1] 소비자동향 + 주가 데이터 분석 시작 ---")
    sp_results_company, model_names_sp, r2_scores_sp = analyze_stock_price(stock_name, consumer_data, df_stock_monthly) # stock_price_analysis.py 에서 분석 실행
    sp_results.update(sp_results_company) # 회사별 주가 예측 결과 통합
    print(f"  --- [분석 1] 소비자동향 + 주가 데이터 분석 완료 ---")


    # --- [분석 2] 소비자동향 + 재무 데이터 분석 ---
    print(f"\n  --- [분석 2] 소비자동향 + 재무 데이터 분석 시작 ---")
    fi_results_company, model_names_fi, r2_scores_fi = analyze_financial_data(stock_name, consumer_data, financial_data) # financial_analysis.py 에서 분석 실행
    fi_results.update(fi_results_company) # 회사별 재무 예측 결과 통합
    plot_model_performance_bar_chart(stock_name, "재무 지표 예측", model_names_fi, r2_scores_fi) # visualization.py 에서 Bar Chart 생성 및 저장
    print(f"  --- [분석 2] 소비자동향 + 재무 데이터 분석 완료 ---")

    print(f"----- {stock_name} 데이터 로드 및 분석 완료 -----")
    print() # 회사별 구분을 위한 빈 줄


print("전체 데이터 로드 및 분석 완료")

# --- 3. 모델 성능 결과 요약 및 출력 ---
print("\n--- [모델 성능 결과 요약] ---")

# 주가 예측 결과 요약 및 CSV 저장
print("\n1. 주가 예측 결과:")
df_sp_results = pd.DataFrame(sp_results).T
print(df_sp_results)
sp_results_csv_filename = "results/stock_price_prediction_results.csv"
df_sp_results.to_csv(sp_results_csv_filename)
print(f"\n주가 예측 결과 CSV 파일 저장 완료: {sp_results_csv_filename}")

# 재무 예측 결과 요약 및 CSV 저장
print("\n2. 재무 예측 결과:")
df_fi_results = pd.DataFrame(fi_results).T
print(df_fi_results)
fi_results_csv_filename = "results/financial_prediction_results.csv"
df_fi_results.to_csv(fi_results_csv_filename)
print(f"재무 예측 결과 CSV 파일 저장 완료: {fi_results_csv_filename}")


# (Option) 딕셔너리에 저장된 데이터 확인 (예시 - 첫 번째 회사 주가, 재무 데이터 head())
if stock_names:
    first_company_name = stock_names[0]
    if stock_data_dict[first_company_name] is not None:
        print(f"\n--- {first_company_name} 주가 데이터 (head): ---")
        print(stock_data_dict[first_company_name].head())
    if financial_data_dict[first_company_name] is not None:
        print(f"\n--- {first_company_name} 재무 데이터 (head, fnclDcdNm, curCd 제외): ---")
        print(financial_data_dict[first_company_name].head())

import numpy as np

# MSE, RMSE 계산 함수 정의
def calculate_mse_rmse(df):
    results = {}
    for col in df.columns[1:]:  # 첫 번째 열(회사명) 제외
        mse = np.mean(df[col] ** 2)  # MSE 계산
        rmse = np.sqrt(mse)  # RMSE 계산
        results[col] = {"MSE": mse, "RMSE": rmse}
    return pd.DataFrame(results).T

# 재무 예측 모델 성능 평가
financial_mse_rmse = calculate_mse_rmse(df_fi_results)

# 주가 예측 모델 성능 평가
stock_mse_rmse = calculate_mse_rmse(df_sp_results)

print(financial_mse_rmse)
print(stock_mse_rmse)