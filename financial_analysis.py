# financial_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 한글 폰트 설정 (financial_analysis.py 에도 추가)
plt.rc('font', family='Malgun Gothic')

def analyze_financial_data(stock_name, consumer_data, financial_data):
    """소비자동향 데이터와 재무 데이터를 분석하여 재무 지표를 예측합니다."""
    fi_results = {} # 재무 예측 결과 저장 딕셔너리
    model_names_fi = ['Linear Regression', 'Random Forest', 'XGBoost', 'ARIMA', 'LSTM'] # 👈 model_names_fi 초기화 (if 블록 밖으로 이동)
    r2_scores_fi = [] # r2_scores_fi 초기화 (if 블록 밖으로 이동)

    if consumer_data is not None and financial_data is not None:
        df_financial = pd.merge(consumer_data, financial_data, on='basDt', how='inner')
        df_financial.fillna(0, inplace=True)
        print("\n  [분석 2] 데이터 병합 완료 (소비자동향 + 재무)")

        # --- Feature Selection ---
        print("  --- Feature Selection 시작 ---")
        # 문제 값 진단 및 처리
        print("    -- 문제 값 진단 및 처리 시작 ---")
        X_financial_fs = df_financial.drop(columns=['basDt', 'enpTastAmt']) # Feature Selection Feature Data (Target 제외, 예시: 자산총계 예측)
        y_financial_fs = df_financial['enpTastAmt'] # Target Data (예시: 자산총계 예측)

        problem_columns_fi = []
        for col in X_financial_fs.columns:
            if np.isinf(X_financial_fs[col]).any() or (X_financial_fs[col] > np.finfo('float32').max).any() or (X_financial_fs[col] < np.finfo('float32').min).any():
                problem_columns_fi.append(col)

        if problem_columns_fi:
            print("    문제 컬럼 발견:", problem_columns_fi)
            print("    문제 컬럼의 기술통계량:")
            print(X_financial_fs[problem_columns_fi].describe())

        X_financial_fs.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대 값 NaN 처리
        print("    무한대 값 NaN 처리 완료")
        X_financial_fs.fillna(0, inplace=True) # NaN 값 0으로 Imputation
        print("    NaN 값 0으로 Imputation 완료")
        X_financial_fs = X_financial_fs.astype(np.float32) # 데이터 타입 float32 변환
        print("    데이터 타입 float32 변환 완료")
        print("    -- 문제 값 진단 및 처리 완료 ---")

        rf_selector_fi = RandomForestRegressor(random_state=42)
        rf_selector_fi.fit(X_financial_fs, y_financial_fs)
        feature_importances_fi = pd.DataFrame({'feature': X_financial_fs.columns, 'importance': rf_selector_fi.feature_importances_})
        feature_importances_fi = feature_importances_fi.sort_values('importance', ascending=False)

        top_n_features_fi = 10
        selected_features_financial = feature_importances_fi['feature'][:top_n_features_fi].tolist()
        X_financial_selected = df_financial[selected_features_financial]

        print(f"    Random Forest Feature Importance 기반 Feature Selection 완료: 상위 {top_n_features_fi}개 Feature 선택")
        print(f"    선택된 Feature: {selected_features_financial}")
        print("  --- Feature Selection 완료 ---")

        # 상관 분석 시각화 및 저장
        plt.figure(figsize=(12, 6))
        sns.heatmap(X_financial_selected.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"[{stock_name}] Selected Feature Correlation Matrix (Consumer Trend + Financial Data)")
        plt.savefig(f"results/{stock_name}_selected_feature_correlation_heatmap_financial.png")
        plt.close()
        print("  [분석 2] Feature Selection 적용 후 상관 분석 히트맵 저장 완료")

        # --- 모델링 및 성능 평가 ---
        print("\n  --- [분석 2] 머신러닝 모델링 및 성능 평가 (재무 지표 예측) 시작 ---")
        financial_target_column = 'enpTastAmt' # 예측 대상 재무 지표
        lr_fi_score = np.nan # 변수 초기화 (NameError 방지)
        rf_fi_score = np.nan
        xgb_fi_score = np.nan
        arima_r2_fi = np.nan
        lstm_r2_fi = np.nan


        if financial_target_column in df_financial.columns: # financial_target_column 존재 여부 확인
            X_financial = df_financial.drop(columns=['basDt', financial_target_column])
            y_financial = df_financial[financial_target_column]
            X_train_fi, X_test_fi, y_train_fi, y_test_fi = train_test_split(X_financial_selected, y_financial, test_size=0.2, random_state=42)

            # 1. Linear Regression
            print("    1. Linear Regression 모델 학습 및 평가 시작")
            lr_fi = LinearRegression()
            lr_fi.fit(X_train_fi, y_train_fi)
            lr_fi_score = lr_fi.score(X_test_fi, y_test_fi)
            print(f"      Linear Regression Score: {lr_fi_score:.4f}")
            r2_scores_fi.append(lr_fi_score)

            # 2. Random Forest
            print("    2. Random Forest 모델 학습 및 평가 시작")
            rf_fi = RandomForestRegressor(random_state=42)
            rf_fi.fit(X_train_fi, y_train_fi)
            rf_fi_score = rf_fi.score(X_test_fi, y_test_fi)
            print(f"      Random Forest Score: {rf_fi_score:.4f}")
            r2_scores_fi.append(rf_fi_score)

            # 3. XGBoost
            print("    3. XGBoost 모델 학습 및 평가 시작")
            xgb_fi = XGBRegressor(random_state=42)
            xgb_fi.fit(X_train_fi, y_train_fi)
            xgb_fi_score = xgb_fi.score(X_test_fi, y_test_fi)
            print(f"      XGBoost Score: {xgb_fi_score:.4f}")
            r2_scores_fi.append(xgb_fi_score)

            # 4. ARIMA
            print("    4. ARIMA 모델 학습 및 평가 시작")
            if financial_data is not None:
                try:
                    train_data_arima_fi = df_financial[:-int(len(df_financial)*0.2)]
                    test_data_arima_fi = df_financial[-int(len(df_financial)*0.2):]
                    model_arima_fi = ARIMA(train_data_arima_fi[financial_target_column], order=(5,1,0))
                    model_arima_fit_fi = model_arima_fi.fit()
                    predictions_arima_fi = model_arima_fit_fi.predict(start=len(train_data_arima_fi), end=len(df_financial)-1)
                    arima_r2_fi = r2_score(test_data_arima_fi[financial_target_column], predictions_arima_fi)
                    print(f"      ARIMA R-squared Score: {arima_r2_fi:.4f}")
                    r2_scores_fi.append(arima_r2_fi)
                except Exception as e:
                    print(f"      ARIMA 모델 학습 중 오류 발생: {e}")
                    arima_r2_fi = np.nan
                    r2_scores_fi.append(arima_r2_fi)
            else:
                print("      ARIMA 모델 학습 생략: 재무 데이터 없음")
                arima_r2_fi = np.nan
                r2_scores_fi.append(arima_r2_fi)

            # 5. LSTM
            print("    5. LSTM 모델 학습 및 평가 시작")
            if financial_data is not None:
                try:
                    train_data_lstm_fi = df_financial[:-int(len(df_financial)*0.2)]
                    test_data_lstm_fi = df_financial[-int(len(df_financial)*0.2):]

                    scaler_lstm_fi = StandardScaler()
                    train_scaled_lstm_fi = scaler_lstm_fi.fit_transform(train_data_lstm_fi[[financial_target_column]])
                    test_scaled_lstm_fi = scaler_lstm_fi.transform(test_data_lstm_fi[[financial_target_column]])

                    def create_sequences(data, seq_length): # 내부 함수로 정의
                        xs = []
                        ys = []
                        for i in range(len(data) - seq_length):
                            x = data[i:(i+seq_length)]
                            y = data[i+seq_length]
                            xs.append(x)
                            ys.append(y)
                        return np.array(xs), np.array(ys)

                    seq_length = 3
                    X_train_lstm_fi, y_train_lstm_fi = create_sequences(train_scaled_lstm_fi, seq_length)
                    X_test_lstm_fi, y_test_fi = create_sequences(test_scaled_lstm_fi, seq_length)

                    X_train_lstm_fi = X_train_lstm_fi.reshape((X_train_lstm_fi.shape[0], X_train_lstm_fi.shape[1], 1))
                    X_test_lstm_fi = X_test_lstm_fi.reshape((X_test_lstm_fi.shape[0], X_test_lstm_fi.shape[1], 1))

                    model_lstm_fi = tf.keras.Sequential([
                        tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(X_train_lstm_fi.shape[1], 1)),
                        tf.keras.layers.Dense(units=1)
                    ])
                    model_lstm_fi.compile(optimizer='adam', loss='mse')
                    model_lstm_fi.fit(X_train_lstm_fi, y_train_lstm_fi, epochs=10, verbose=0)

                    predictions_lstm_scaled_fi = model_lstm_fi.predict(X_test_lstm_fi, verbose=0)
                    predictions_lstm_fi = scaler_lstm_fi.inverse_transform(predictions_lstm_scaled_fi)
                    lstm_mse_fi = mean_squared_error(test_data_lstm_fi[[financial_target_column]], predictions_lstm_fi)
                    lstm_r2_fi = r2_score(test_data_lstm_fi[[financial_target_column]], predictions_lstm_fi)
                    print(f"      LSTM R-squared Score: {lstm_r2_fi:.4f}, MSE: {lstm_mse_fi:.4f}")
                    r2_scores_fi.append(lstm_r2_fi)
                    print(f"      [디버깅] LSTM X_train shape: {X_train_lstm_fi.shape}")  # financial_analysis.py 재무 예측
                    print(f"      [디버깅] LSTM X_test shape: {X_test_lstm_fi.shape}")  # financial_analysis.py 재무 예측

                except Exception as e:
                    print(f"      LSTM 모델 학습 중 오류 발생: {e}")
                    lstm_r2_fi = np.nan
                    lstm_mse_fi = np.nan
                    r2_scores_fi.append(lstm_r2_fi)
            else:
                print("      LSTM 모델 학습 생략: 재무 데이터 없음")
                lstm_r2_fi = np.nan
                r2_scores_fi.append(lstm_r2_fi)


            print(f"\n  [분석 2] Feature Selection 적용 후 머신러닝 모델 성능 (재무 지표 예측 - {financial_target_column}):")
            print(f"    Linear Regression Score: {lr_fi_score:.4f}")
            print(f"    Random Forest Score: {rf_fi_score:.4f}")
            print(f"    XGBoost Score: {xgb_fi_score:.4f}")
            print(f"    ARIMA R-squared Score: {arima_r2_fi:.4f}")
            print(f"    LSTM R-squared Score: {lstm_r2_fi:.4f}, MSE: {lstm_mse_fi:.4f}")
            print("  --- [분석 2] 머신러닝 모델링 및 성능 평가 (재무 지표 예측) 완료 ---")

            fi_results[stock_name] = { # 딕셔너리에 결과 저장
                'Linear Regression': lr_fi_score,
                'Random Forest': rf_fi_score,
                'XGBoost': xgb_fi_score,
                'ARIMA': arima_r2_fi,
                'LSTM': lstm_r2_fi
            }
        else:
             print(f"  [분석 2] 머신러닝 모델링 생략: '{financial_target_column}' 컬럼이 df_financial에 없음") # financial_target_column 없을 경우 메시지 추가

    else:
        print("  [분석 2] 소비자동향 + 재무 데이터 분석 생략 (데이터 로드 실패)")
    print(f"  --- [분석 2] 소비자동향 + 재무 데이터 분석 완료 ---")
    return fi_results, model_names_fi, r2_scores_fi

# (main.py에서 호출 예시)
# from financial_analysis import analyze_financial_data
# fi_results, model_names_fi, r2_scores_fi = analyze_financial_data(stock_name, consumer_data, financial_data)