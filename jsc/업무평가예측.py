"""
performance_model.py
- CSV로 업무평가(점수 또는 등급) 예측
- 전처리(결측치/스케일/원-핫), 학습/검증, 중요도(퍼뮤테이션) 시각화, 모델 저장/불러오기, 사용자 입력 예측까지 포함
- 사용법:
  1) 학습:
     python performance_model.py train --csv_path data/employee.csv --target performance_score --model_path artifacts/model.joblib --imp_png artifacts/feature_importance.png
  2) 예측(저장된 모델로):
     python performance_model.py predict --model_path artifacts/model.joblib --json '{"age":33,"travel_allowance_per_day":35000,"department":"Sales","overtime_hours":10,"business_trip_days":4,"daily_allowance":20000}'
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 🧱 유틸 함수들
# =========================

def detect_problem_type(y: pd.Series) -> str:
    """
    타깃의 dtype/고유값을 보고 문제 유형 자동 판별
    - 숫자형 + 고유값이 많으면 회귀
    - 그 외는 분류
    """
    if pd.api.types.is_numeric_dtype(y):
        # 숫자형인데 유니크 값이 적으면 등급형 점수로 분류일 수도 있음
        n_unique = y.nunique(dropna=True)
        if n_unique <= 10:
            return "classification"
        return "regression"
    else:
        return "classification"


def split_features(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    """
    데이터프레임에서 숫자/범주형 컬럼 자동 분리 (타깃 제외)
    """
    feature_cols = [c for c in df.columns if c != target]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    return num_cols, cat_cols


def build_preprocess(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    숫자/범주형 전처리 파이프라인 구성
    - 숫자: 결측치 평균 대치 + 표준화
    - 범주: 결측치 'missing' 대치 + 원-핫 인코딩
    """
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )
    return pre


def pick_model(problem_type: str):
    """
    문제 유형별 기본 모델 선택
    - 회귀: RandomForestRegressor
    - 분류: RandomForestClassifier
    """
    if problem_type == "regression":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    else:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )


def evaluate_classification(y_true, y_pred, y_proba=None, label="TEST"):
    """
    분류 성능 지표 출력
    """
    print(f"\n===== 📊 Classification Metrics ({label}) =====")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1 (weighted):", round(f1_score(y_true, y_pred, average="weighted"), 4))
    try:
        # 이진 분류일 때만 ROC-AUC 시도
        if y_proba is not None and y_proba.shape[1] == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
            print("ROC-AUC:", round(auc, 4))
    except Exception:
        pass
    print("\nClassification report:\n", classification_report(y_true, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def evaluate_regression(y_true, y_pred, label="TEST"):
    """
    회귀 성능 지표 출력
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"\n===== 📈 Regression Metrics ({label}) =====")
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


def plot_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    top_k: int = 20,
    save_png: str | None = None,
    feature_names: List[str] | None = None
):
    """
    퍼뮤테이션 중요도(모델 불가지론적) 계산 및 막대 그래프 저장
    - 전처리 포함 파이프라인에 대해 안전하게 사용 가능
    """
    print("\n🔍 Calculating permutation importance (this can take a while)...")
    scoring = "neg_mean_squared_error" if problem_type == "regression" else "f1_weighted"
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring=scoring
    )
    importances = result.importances_mean
    stds = result.importances_std

    # 중요도 상위 정렬
    idx = np.argsort(importances)[::-1]
    idx = idx[:top_k]

    # 특성 이름 추적: ColumnTransformer 이후 특성명 복원
    if feature_names is None:
        try:
            ohe = pipeline.named_steps["model"].__class__.__name__  # dummy to use try
        except Exception:
            pass

    # 실제로는 전처리 변환 후 칼럼명이 바뀌지만, 여기선 원본 피처 단위 중요도를 보는 용도로
    # permutation_importance는 입력 X_test 기준으로 피처를 섞으므로 원본 칼럼명 사용 가능
    plotted_names = list(X_test.columns)

    plt.figure(figsize=(9, max(4, int(len(idx) * 0.4))))
    plt.barh(range(len(idx)), importances[idx], xerr=stds[idx])
    plt.yticks(range(len(idx)), [plotted_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.title("Permutation Importance (higher = more important)")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_png:
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        plt.savefig(save_png, dpi=160)
        print(f"💾 중요도 그래프 저장: {save_png}")
    else:
        plt.show()
    plt.close()


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, problem_type: str):
    """
    빠른 5-Fold 교차검증 점수 확인
    """
    if problem_type == "regression":
        scoring = "r2"
    else:
        scoring = "f1_weighted"
    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scoring, n_jobs=-1)
    print(f"\n🔁 5-Fold CV ({scoring}) -> mean={scores.mean():.4f} ± {scores.std():.4f}")


# =========================
# 🧠 학습 함수 (end-to-end)
# =========================

def train(
    csv_path: str,
    target: str = "performance_score",
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = "artifacts/model.joblib",
    importance_png: str | None = "artifacts/feature_importance.png"
):
    """
    CSV로부터 학습 → 검증 → 중요도 저장 → 파이프라인 저장
    """
    print(f"📥 Load CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise ValueError(f"타깃 컬럼 '{target}' 이(가) CSV에 없습니다. 컬럼 목록: {list(df.columns)}")

    # 타깃/피처 분리
    y = df[target]
    X = df.drop(columns=[target])

    # 문제 유형 결정
    problem_type = detect_problem_type(y)
    print(f"🔎 Detected problem type: {problem_type}")

    # 숫자/범주형 분할
    num_cols, cat_cols = split_features(df, target)

    # 전처리 + 모델 파이프라인
    pre = build_preprocess(num_cols, cat_cols)
    model = pick_model(problem_type)

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("model", model)
    ])

    # 학습/검증 분리 (분류면 stratify)
    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # 학습
    print("\n🚀 Training...")
    pipe.fit(X_train, y_train)

    # 평가
    print("\n✅ Evaluation on TEST:")
    y_pred = pipe.predict(X_test)

    if problem_type == "classification":
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None
        evaluate_classification(y_test, y_pred, y_proba, label="TEST")
    else:
        evaluate_regression(y_test, y_pred, label="TEST")

    # 교차검증
    cross_validate(pipe, X, y, problem_type)

    # 중요도 저장 (원본 X_test 칼럼명 사용)
    plot_permutation_importance(
        pipeline=pipe,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type,
        save_png=importance_png,
        feature_names=list(X.columns)
    )

    # 파이프라인 + 메타정보 저장
    meta = {
        "problem_type": problem_type,
        "target": target,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_columns": list(X.columns)
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    return pipe, meta


# =========================
# 🔮 예측 함수
# =========================

def predict_with_saved_model(model_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    저장된 파이프라인으로 단건 예측
    - payload: 사용자 입력 딕셔너리 (CSV의 피처 컬럼명과 동일해야 함)
    """
    bundle = joblib.load(model_path)
    pipe: Pipeline = bundle["pipeline"]
    meta = bundle["meta"]

    # 입력 유효성 체크 및 DataFrame 변환
    X_cols = meta["feature_columns"]
    x_df = pd.DataFrame([payload], columns=X_cols)  # 누락 컬럼은 NaN으로 들어감

    # 예측
    y_pred = pipe.predict(x_df)

    result = {
        "problem_type": meta["problem_type"],
        "prediction": y_pred[0]
    }

    # 분류일 경우 확률도 제공
    if meta["problem_type"] == "classification":
        try:
            proba = pipe.predict_proba(x_df)[0]
            classes = pipe.named_steps["model"].classes_
            result["proba"] = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception:
            pass

    return result


# =========================
# 🖥️ CLI 진입점
# =========================

def main():
    parser = argparse.ArgumentParser(description="Work Performance Prediction (Regression/Classification auto)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a model from CSV")
    p_train.add_argument("--csv_path", required=True, type=str)
    p_train.add_argument("--target", default="performance_score", type=str)
    p_train.add_argument("--test_size", default=0.2, type=float)
    p_train.add_argument("--random_state", default=42, type=int)
    p_train.add_argument("--model_path", default="artifacts/model.joblib", type=str)
    p_train.add_argument("--imp_png", default="artifacts/feature_importance.png", type=str)

    # predict
    p_pred = sub.add_parser("predict", help="Predict with saved model")
    p_pred.add_argument("--model_path", required=True, type=str)
    p_pred.add_argument("--json", required=True, type=str, help="JSON string of feature values")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            csv_path=args.csv_path,
            target=args.target,
            test_size=args.test_size,
            random_state=args.random_state,
            model_path=args.model_path,
            importance_png=args.imp_png
        )

    elif args.cmd == "predict":
        payload = json.loads(args.json)
        res = predict_with_saved_model(args.model_path, payload)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
