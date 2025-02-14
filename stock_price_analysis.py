# stock_price_analysis.py
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

# 한글 폰트 설정 (stock_price_analysis.py 에도 추가)
plt.rc('font', family='Malgun Gothic')

def analyze_stock_price(stock_name, consumer_data, df_stock_monthly):
    """소비자동향 데이터와 주가 데이터를 분석하여 주가를 예측합니다."""
    sp_results = {}  # 주가 예측 결과 저장 딕셔너리
    model_names_sp = ['Linear Regression', 'Random Forest', 'XGBoost', 'ARIMA','LSTM']  # 👈 model_names_sp 초기화 (if 블록 밖으로 이동)
    r2_scores_sp = []  # r2_scores_sp 초기화 (if 블록 밖으로 이동)

    if consumer_data is not None and df_stock_monthly is not None:
        df_stock_price = pd.merge(consumer_data, df_stock_monthly, on='basDt', how='inner')
        df_stock_price.fillna(0, inplace=True)
        print("\n  [분석 1] 데이터 병합 완료 (소비자동향 + 주가)")

        # --- Feature Selection ---
        print("  --- Feature Selection 시작 ---")
        X_stock_price_fs = df_stock_price.drop(columns=['basDt', 'clpr']) # Feature Selection Feature Data (Target 제외)
        y_stock_price_fs = df_stock_price['clpr'] # Target Data

        rf_selector = RandomForestRegressor(random_state=42)
        rf_selector.fit(X_stock_price_fs, y_stock_price_fs)
        feature_importances = pd.DataFrame({'feature': X_stock_price_fs.columns, 'importance': rf_selector.feature_importances_})
        feature_importances = feature_importances.sort_values('importance', ascending=False)

        top_n_features = 10
        selected_features_stock_price = feature_importances['feature'][:top_n_features].tolist()
        X_stock_price_selected = df_stock_price[selected_features_stock_price]

        print(f"    Random Forest Feature Importance 기반 Feature Selection 완료: 상위 {top_n_features}개 Feature 선택")
        print(f"    선택된 Feature: {selected_features_stock_price}")
        print("  --- Feature Selection 완료 ---")

        # 상관 분석 시각화 및 저장
        plt.figure(figsize=(12, 6))
        sns.heatmap(X_stock_price_selected.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"[{stock_name}] Selected Feature Correlation Matrix (Consumer Trend + Stock Price)")
        plt.savefig(f"results/{stock_name}_selected_feature_correlation_heatmap_stock_price.png")
        plt.close()
        print("  [분석 1] Feature Selection 적용 후 상관 분석 히트맵 저장 완료")

        # --- 모델링 및 성능 평가 ---
        print("\n  --- [분석 1] 머신러닝 모델링 및 성능 평가 (주가 예측) 시작 ---")
        X_stock_price = df_stock_price.drop(columns=['basDt', 'clpr'])
        y_stock_price = df_stock_price['clpr']
        X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_stock_price_selected, y_stock_price, test_size=0.2, random_state=42)

        # 1. Linear Regression
        print("    1. Linear Regression 모델 학습 및 평가 시작")
        lr_sp = LinearRegression()
        lr_sp.fit(X_train_sp, y_train_sp)
        lr_sp_score = lr_sp.score(X_test_sp, y_test_sp)
        print(f"      Linear Regression Score: {lr_sp_score:.4f}")
        r2_scores_sp.append(lr_sp_score)

        # 2. Random Forest
        print("    2. Random Forest 모델 학습 및 평가 시작")
        rf_sp = RandomForestRegressor(random_state=42)
        rf_sp.fit(X_train_sp, y_train_sp)
        rf_sp_score = rf_sp.score(X_test_sp, y_test_sp)
        print(f"      Random Forest Score: {rf_sp_score:.4f}")
        r2_scores_sp.append(rf_sp_score)

        # 3. XGBoost
        print("    3. XGBoost 모델 학습 및 평가 시작")
        xgb_sp = XGBRegressor(random_state=42)
        xgb_sp.fit(X_train_sp, y_train_sp)
        xgb_sp_score = xgb_sp.score(X_test_sp, y_test_sp)
        print(f"      XGBoost Score: {xgb_sp_score:.4f}")
        r2_scores_sp.append(xgb_sp_score)

        # 4. ARIMA
        print("    4. ARIMA 모델 학습 및 평가 시작")
        if df_stock_monthly is not None:
            try:
                train_data_arima_sp = df_stock_monthly[:-int(len(df_stock_monthly)*0.2)]
                test_data_arima_sp = df_stock_monthly[-int(len(df_stock_monthly)*0.2):]
                model_arima_sp = ARIMA(train_data_arima_sp['clpr'], order=(5,1,0))
                model_arima_fit_sp = model_arima_sp.fit()
                predictions_arima_sp = model_arima_fit_sp.predict(start=len(train_data_arima_sp), end=len(df_stock_monthly)-1)
                arima_r2_sp = r2_score(test_data_arima_sp['clpr'], predictions_arima_sp)
                print(f"      ARIMA R-squared Score: {arima_r2_sp:.4f}")
                r2_scores_sp.append(arima_r2_sp)
            except Exception as e:
                print(f"      ARIMA 모델 학습 중 오류 발생: {e}")
                arima_r2_sp = np.nan
                r2_scores_sp.append(arima_r2_sp)
        else:
            print("      ARIMA 모델 학습 생략: 월별 주가 데이터 없음")
            arima_r2_sp = np.nan
            r2_scores_sp.append(arima_r2_sp)

        # 5. LSTM
        print("    5. LSTM 모델 학습 및 평가 시작")
        if df_stock_monthly is not None:
            try:
                train_data_lstm_sp = df_stock_monthly[:-int(len(df_stock_monthly)*0.2)]
                test_data_lstm_sp = df_stock_monthly[-int(len(df_stock_monthly)*0.2):]

                scaler_lstm_sp = StandardScaler()
                train_scaled_lstm_sp = scaler_lstm_sp.fit_transform(train_data_lstm_sp[['clpr']])
                test_scaled_lstm_sp = scaler_lstm_sp.transform(test_data_lstm_sp[['clpr']])

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
                X_train_lstm_sp, y_train_lstm_sp = create_sequences(train_scaled_lstm_sp, seq_length)
                X_test_lstm_sp, y_test_lstm_sp = create_sequences(test_scaled_lstm_sp, seq_length)

                X_train_lstm_sp = X_train_lstm_sp.reshape((X_train_lstm_sp.shape[0], X_train_lstm_sp.shape[1], 1))
                X_test_lstm_sp = X_test_lstm_sp.reshape((X_test_lstm_sp.shape[0], X_test_lstm_sp.shape[1], 1))

                model_lstm_sp = tf.keras.Sequential([
                    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(X_train_lstm_sp.shape[1], 1)),
                    tf.keras.layers.Dense(units=1)
                ])
                model_lstm_sp.compile(optimizer='adam', loss='mse')
                model_lstm_sp.fit(X_train_lstm_sp, y_train_lstm_sp, epochs=10, verbose=0)

                predictions_lstm_scaled_sp = model_lstm_sp.predict(X_test_lstm_sp, verbose=0)
                predictions_lstm_sp = scaler_lstm_sp.inverse_transform(predictions_lstm_scaled_sp)
                lstm_mse_sp = mean_squared_error(test_data_lstm_sp[['clpr']], predictions_lstm_sp)
                lstm_r2_sp = r2_score(test_data_lstm_sp[['clpr']], predictions_lstm_sp)
                print(f"      LSTM R-squared Score: {lstm_r2_sp:.4f}, MSE: {lstm_mse_sp:.4f}")
                r2_scores_sp.append(lstm_r2_sp)
                print(f"      [디버깅] LSTM X_train shape: {X_train_lstm_sp.shape}")  # stock_price_analysis.py 주가 예측
                print(f"      [디버깅] LSTM X_test shape: {X_test_lstm_sp.shape}")  # stock_price_analysis.py 주가 예측
            except Exception as e:
                print(f"      LSTM 모델 학습 중 오류 발생: {e}")
                lstm_r2_sp = np.nan
                lstm_mse_sp = np.nan
                r2_scores_sp.append(lstm_r2_sp)
        else:
            print("      LSTM 모델 학습 생략: 월별 주가 데이터 없음")
            lstm_r2_sp = np.nan
            lstm_mse_sp = np.nan
            r2_scores_sp.append(lstm_r2_sp)

        print("\n  [분석 1] Feature Selection 적용 후 머신러닝 모델 성능 (주가 예측):")
        print(f"    Linear Regression Score: {lr_sp_score:.4f}")  # ✅ 올바른 변수명: lr_sp_score
        print(f"    Random Forest Score: {rf_sp_score:.4f}")  # ✅ 수정: r_sp_score -> rf_sp_score (RandomForest)
        print(f"    XGBoost Score: {xgb_sp_score:.4f}")
        print(f"    ARIMA R-squared Score: {arima_r2_sp:.4f}")
        print(f"    LSTM R-squared Score: {lstm_r2_sp:.4f}, MSE: {lstm_mse_sp:.4f}")
        print("  --- [분석 1] 머신러닝 모델링 및 성능 평가 (주가 예측) 완료 ---")

        sp_results[stock_name] = { # 딕셔너리에 결과 저장
            'Linear Regression': lr_sp_score,
            'Random Forest': rf_sp_score,
            'XGBoost': xgb_sp_score,
            'ARIMA': arima_r2_sp,
            'LSTM': lstm_r2_sp
        }
    else:
        print("  [분석 1] 소비자동향 + 주가 데이터 분석 생략 (데이터 로드 실패)")
    print(f"  --- [분석 1] 소비자동향 + 주가 데이터 분석 완료 ---")
    return sp_results, model_names_sp, r2_scores_sp

# (main.py에서 호출 예시)
# from stock_price_analysis import analyze_stock_price
# sp_results, model_names_sp, r2_scores_sp = analyze_stock_price(stock_name, consumer_data, df_stock_monthly)