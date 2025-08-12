# -*- coding: utf-8 -*-
# 파일명 예: train_decision_tree2_nosmote.py
# 목적: 업그레이드/버전 충돌 없이 돌아가는 결정트리 + GridSearchCV (f1_macro 최적화)
# 의존성: pip install pandas scikit-learn matplotlib joblib

import os, sys, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, log_loss, classification_report,
                             confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# ----------------------- 경로/설정 -----------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_NAME   = "WA_Fn-UseC_-HR-Employee-Attrition_변환.csv"   # 필요 시 수정
CSV_PATH   = os.path.join(BASE_DIR, CSV_NAME)
TARGET_COL = "업무평가"
OUTDIR     = os.path.join(BASE_DIR, "output2")
MODEL_PATH = os.path.join(OUTDIR, "performance_tree_pipeline_tuned.pkl")
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
    if drop_cols: df = df.drop(columns=drop_cols)

    # 3) 결측치 제거(간단)
    df = df.dropna(axis=0)

    # 4) 분리
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    if y.dtype == "O":
        y = y.astype("category").cat.codes

    # 5) 전처리자(희소X, 밀집 출력: 일부 환경에서 호환성↑)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)  # sklearn>=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)         # 하위버전 대응

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", "passthrough", num_cols),
    ])

    # 6) 학습/검증 분리(가능하면 계층)
    cls_counts = pd.Series(y).value_counts()
    use_strat = (len(cls_counts) > 1) and (cls_counts.min() >= 2)
    if not use_strat:
        print("[안내] 소수 클래스 표본이 2 미만 → stratify 미사용")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y if use_strat else None)
    )

    # 7) 파이프라인: 전처리 → 결정트리 (클래스 불균형은 class_weight로 보정)
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])

    # 8) 하이퍼파라미터 탐색(f1_macro 기준)
    param_grid = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [3, 4, 5, 6, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005, 0.01],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # n_jobs=1로 고정 → threadpool 관련 충돌/제한 회피
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring="f1_macro",
        cv=cv, n_jobs=1, refit=True, verbose=0
    )

    # 9) 학습
    gs.fit(Xtr, ytr)
    best_pipe = gs.best_estimator_
    print(f"\n[튜닝] Best params: {gs.best_params_}")
    print(f"[튜닝] CV best f1_macro: {gs.best_score_:.4f}")

    # 10) 평가
    yp = best_pipe.predict(Xte)
    yp_prob = best_pipe.predict_proba(Xte)

    acc  = accuracy_score(yte, yp)
    bacc = balanced_accuracy_score(yte, yp)
    prec = precision_score(yte, yp, average="macro",  zero_division=0)
    rec  = recall_score(yte, yp,  average="macro",  zero_division=0)
    f1   = f1_score(yte, yp,     average="macro",  zero_division=0)
    loss = log_loss(yte, yp_prob)

    print("\n=== 성능지표 (결정트리 튜닝, macro 평균) ===")
    print(f"loss        : {loss:.3f}")
    print(f"accuracy    : {acc:.2%}")
    print(f"precision   : {prec:.3f}")
    print(f"recall      : {rec:.3f}")
    print(f"f1_score    : {f1:.3f}")
    print(f"balanced_acc: {bacc:.2%}")

    print("\n=== 분류 리포트 ===")
    print(classification_report(yte, yp, digits=4))
    print("=== 혼동행렬 ===")
    print(confusion_matrix(yte, yp))

    # 11) 결과 저장
    os.makedirs(OUTDIR, exist_ok=True)

    # 전처리 후 특성명
    ohe_fitted = best_pipe.named_steps["prep"].named_transformers_["cat"]
    ohe_names  = ohe_fitted.get_feature_names_out(cat_cols).tolist() if cat_cols else []
    feature_names = ohe_names + num_cols

    # 중요도 저장
    dt = best_pipe.named_steps["model"]
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": dt.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTDIR, "feature_importances_performance_tuned.csv"), index=False, encoding="utf-8-sig")

    # 해석용 얕은 트리(depth=3)
    prep = best_pipe.named_steps["prep"]
    Xtr_enc = prep.fit_transform(Xtr)

    shallow_dt = clone(dt)
    shallow_dt.set_params(max_depth=3)
    shallow_dt.fit(Xtr_enc, ytr)

    plt.figure(figsize=(24, 14))
    plot_tree(shallow_dt, feature_names=feature_names,
              class_names=[str(c) for c in sorted(pd.Series(ytr).unique())],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree (tuned, depth=3, human-readable)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "decision_tree_performance_depth3_tuned.png"), dpi=200)
    plt.close()

    # 원본(best 파라미터) 트리
    plt.figure(figsize=(28, 16))
    plot_tree(dt, feature_names=feature_names,
              class_names=[str(c) for c in sorted(pd.Series(ytr).unique())],
              filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree (tuned, full depth)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "decision_tree_performance_full_tuned.png"), dpi=200)
    plt.close()

    # 모델 저장
    joblib.dump(best_pipe, MODEL_PATH)
    print("\n[완료] output2 폴더에 저장됨:")
    print(" - feature_importances_performance_tuned.csv")
    print(" - decision_tree_performance_depth3_tuned.png")
    print(" - decision_tree_performance_full_tuned.png")
    print(f" - performance_tree_pipeline_tuned.pkl  ({MODEL_PATH})")

if __name__ == "__main__":
    main()
