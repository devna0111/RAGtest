# -*- coding: utf-8 -*-
# 목적: CSV로 '업무평가'를 예측하는 결정트리 모델 (학습 + 평가 + 시각화 + 저장)
# 의존성: pip install pandas scikit-learn matplotlib joblib

import os, sys, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, log_loss,
    classification_report, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ----------------------- 경로/설정 -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition_변환.csv"   # 파일명이 다르면 바꾸세요
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)
TARGET_COL = "업무평가"
OUTDIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUTDIR, "performance_tree_pipeline.pkl")
# --------------------------------------------------------

def read_csv_safely(path):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def main():
    # 1) 데이터 로드
    if not os.path.exists(CSV_PATH):
        print(f"[에러] CSV를 찾을 수 없습니다: {CSV_PATH}")
        sys.exit(1)
    df = read_csv_safely(CSV_PATH)

    if TARGET_COL not in df.columns:
        print(f"[에러] '{TARGET_COL}' 컬럼이 없습니다. 현재 컬럼 예시: {list(df.columns)[:20]}")
        sys.exit(1)
    print(f"[정보] 타깃 컬럼: {TARGET_COL}")

    # 2) 불필요 컬럼 제거(있으면)
    drop_cols = [c for c in ["EmployeeNumber","EmployeeID","ID","Over18","StandardHours","EmployeeCount"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 3) 결측치 간단 처리
    df = df.dropna(axis=0)

    # 4) X / y 분리
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    if y.dtype == "O":
        y = y.astype("category").cat.codes  # 문자열 타깃이면 코드화

    # 5) 전처리 파이프라인
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    # 6) 학습/검증 분할 (계층분할은 소수 클래스 2개 이상일 때만)
    cls_counts = pd.Series(y).value_counts()
    use_strat = (len(cls_counts) > 1) and (cls_counts.min() >= 2)
    if not use_strat:
        print("[안내] 소수 클래스 표본이 2 미만 → stratify 미사용")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y if use_strat else None)
    )

    # 7) 모델 구성/학습
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,              # 필요 시 조정
        min_samples_split=10,     # 필요 시 조정
        class_weight="balanced",  # 불균형 보정
        random_state=42
    )
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(Xtr, ytr)

    # 8) 평가 (★ 요청하신 지표 포함)
    yp = pipe.predict(Xte)
    yp_prob = pipe.predict_proba(Xte)  # log_loss 계산용

    acc  = accuracy_score(yte, yp)
    bacc = balanced_accuracy_score(yte, yp)
    prec = precision_score(yte, yp, average="weighted", zero_division=0)
    rec  = recall_score(yte, yp, average="weighted", zero_division=0)
    f1   = f1_score(yte, yp, average="weighted", zero_division=0)
    loss = log_loss(yte, yp_prob)

    print("\n=== 성능지표 (업무평가 분류) ===")
    print(f"loss       : {loss:.3f}")
    print(f"accuracy   : {acc:.2%}")
    print(f"precision  : {prec:.2f}")
    print(f"recall     : {rec:.2f}")
    print(f"f1_score   : {f1:.2f}")
    print(f"balanced_acc : {bacc:.2%}")

    print("\n=== 분류 리포트 ===")
    print(classification_report(yte, yp, digits=4))
    print("=== 혼동행렬 ===")
    print(confusion_matrix(yte, yp))

    # 9) 피처 중요도 저장
    os.makedirs(OUTDIR, exist_ok=True)
    ohe = pipe.named_steps["prep"].named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    feature_names = ohe_names + num_cols
    importances = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTDIR, "feature_importances_performance.csv"), index=False)

    # 10) 시각화(해석용 depth=3 + 원본 depth=5)
    # 해석용 얕은 트리
    shallow = Pipeline([
        ("prep", preprocess),
        ("model", DecisionTreeClassifier(max_depth=3, min_samples_split=10, class_weight="balanced", random_state=42))
    ])
    shallow.fit(Xtr, ytr)
    plt.figure(figsize=(24, 14))
    plot_tree(shallow.named_steps["model"],
              feature_names=feature_names,
              class_names=[str(c) for c in sorted(pd.Series(ytr).unique())],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree for Performance (depth=3, human-readable)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "decision_tree_performance_depth3.png"), dpi=200)
    plt.close()

    # 원본 트리
    plt.figure(figsize=(28, 16))
    plot_tree(pipe.named_steps["model"],
              feature_names=feature_names,
              class_names=[str(c) for c in sorted(pd.Series(ytr).unique())],
              filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree for Performance (depth=5)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "decision_tree_performance_depth5.png"), dpi=200)
    plt.close()

    # 11) 모델 저장
    joblib.dump(pipe, MODEL_PATH)
    print("\n[완료] outputs 폴더에 저장됨:")
    print(" - feature_importances_performance.csv")
    print(" - decision_tree_performance_depth3.png")
    print(" - decision_tree_performance_depth5.png")
    print(f" - performance_tree_pipeline.pkl  ({MODEL_PATH})")

if __name__ == "__main__":
    main()
