# visualization.py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 한글 폰트 설정 (visualization.py 에도 추가)
plt.rc('font', family='Malgun Gothic')

def plot_model_performance_bar_chart(stock_name, analysis_type, model_names, r2_scores):
    """모델 성능 막대 그래프를 생성하고 저장합니다."""
    plt.figure(figsize=(10, 6))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightseagreen', 'plum'] # 모델별 색상 지정

    if not r2_scores: # r2_scores가 비어있는 경우 체크
        print("R-squared 점수 데이터가 없습니다. 막대 그래프를 그릴 수 없습니다.")
        plt.text(0.5, 0.5, 'R-squared 점수 데이터가 없습니다.', ha='center', va='center', fontsize=12) # 그래프 내에 텍스트 표시
    else:
        bars = plt.bar(model_names, r2_scores, color=colors) # 색상 적용

        # y 축 formatter (R-squared 값 %로 표시)
        formatter = mticker.PercentFormatter(xmax=1.0)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.ylim([-1, 1]) # y축 범위 조정 (R-squared는 -1 ~ 1 또는 그 이상)
        plt.xlabel('모델')
        plt.ylabel('R-squared')
        title = f"[{stock_name}] 모델별 성능 비교 (R-squared) - {analysis_type}" # 분석 타입(주가/재무) 제목에 추가
        plt.title(title)

        # bar 위에 R-squared 값 텍스트로 표시
        for bar in bars:
            height = bar.get_height()
            label_position_y = height + 0.02 if height >= 0 else height - 0.07 # label position 조정
            plt.text(bar.get_x() + bar.get_width() / 2, label_position_y,
                     f'{height:.2%}', ha='center', va='bottom') # format: percent

        plt.xticks(rotation=45, ha='right') # x축 label 회전
        plt.tight_layout() # 레이아웃 조정


    filename = f"results/{stock_name}_model_performance_bar_chart_{analysis_type}.png" # 파일명에 분석 타입 추가
    plt.savefig(filename)
    plt.close()
    print(f"    모델 성능 Bar Chart 저장 완료: {filename}")

# (main.py에서 호출 예시)
# from visualization import plot_model_performance_bar_chart
# plot_model_performance_bar_chart(stock_name, "주가 예측", model_names_sp, r2_scores_sp)