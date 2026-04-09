import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def run_training():
    # 1. Đọc file CSV 41 mẫu
    df = pd.read_csv('features_extraction/final_features.csv')

    # 2. Tách đặc trưng (X) và nhãn (y)
    X = df[['Hz_Peaks', 'Hz_FFT', 'Amplitude_mm']]
    y = df['Label']

    # 3. Chia 80% để học, 20% để thi (khoảng 8 mẫu đi thi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Chuẩn hóa dữ liệu (Đưa Hz và mm về cùng một hệ quy chiếu)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Huấn luyện mô hình XGBoost
    model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)
    model.fit(X_train_scaled, y_train)

    # 6. Đánh giá kết quả sơ bộ
    y_pred = model.predict(X_test_scaled)
    print("\n" + "=" * 30)
    print(f"ĐỘ CHÍNH XÁC SƠ BỘ: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("=" * 30)
    print(classification_report(y_test, y_pred))

    # 7. Lưu "Bộ não" AI lại để tuần sau dùng dự đoán thực tế
    with open('parkinson_model.pkl', 'wb') as f: pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    print("\nĐã lưu mô hình thành công!")


if __name__ == "__main__":
    run_training()