# -*- coding: utf-8 -*-
# 목적: 여러 분류기를 대상으로 '정확도(accuracy)'를 최대화하는 모델을 자동 선택/저장
# 의존성: pandas, scikit-learn, joblib (matplotlib는 선택)
# 실행: python train_models_accuracy.py

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

# ----------------------------- 경로/설정 -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "WA_Fn-UseC_-HR-Employee-Attrition_변환.csv"
CSV_PATH = os.path.join(BASE_DIR, CSV_NAME)
TARGET_COL = "업무평가"
OUTDIR = os.path.join(BASE_DIR, "output3")  # ← 변경됨
BEST_MODEL_PATH = os.path.join(OUTDIR, "best_accuracy_model.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
# -------------------------------------------------------------------

def read_csv_safely(path):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[에러] CSV를 찾을 수 없습니다: {CSV_PATH}")
        sys.exit(1)
    df = read_csv_safely(CSV_PATH)

    if TARGET_COL not in df.columns:
        print(f"[에러] 타깃 컬럼 '{TARGET_COL}' 이(가) 없습니다. 현재 컬럼: {list(df.columns)}")
        sys.exit(1)

    drop_cols = [c for c in ["EmployeeNumber","EmployeeID","ID","Over18","StandardHours","EmployeeCount"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.dropna(axis=0)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    if y.dtype == "O":
        y = y.astype("category").cat.codes

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)
    scaler = StandardScaler(with_mean=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", scaler, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    cls_counts = pd.Series(y).value_counts()
    use_strat = (len(cls_counts) > 1) and (cls_counts.min() >= 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=(y if use_strat else None)
    )

    models_and_params = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__criterion": ["gini", "entropy", "log_loss"],
                "model__max_depth": [None, 4, 6, 8, 12],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, n_jobs=1),
            {
                "model__C": [0.1, 1, 3, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            },
        ),
        "SVC": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1, 3, 10],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_score, best_est, all_cv_rows = None, -1, None, []

    for name, (model, param_grid) in models_and_params.items():
        pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=1,
            refit=True,
            verbose=0,
            return_train_score=False,
        )
        print(f"\n[탐색] {name} 시작...")
        gs.fit(X_train, y_train)

        print(f"[탐색] {name} best accuracy (CV): {gs.best_score_:.4f}")
        print(f"[탐색] {name} best params: {gs.best_params_}")

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.insert(0, "model", name)
        all_cv_rows.append(tmp)

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_est = gs.best_estimator_
            best_name = name

    cv_df = pd.concat(all_cv_rows, ignore_index=True)
    cv_path = os.path.join(OUTDIR, "cv_results_accuracy_all_models.csv")
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
    print(f"\n[저장] CV 결과: {cv_path}")

    if best_est is None:
        print("[에러] 최적 모델이 없습니다.")
        sys.exit(1)

    print(f"\n[선정] 최적 모델: {best_name}  |  CV best accuracy: {best_score:.4f}")

    y_pred = best_est.predict(X_test)
    y_prob = best_est.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        ll = log_loss(y_test, y_prob)
    except Exception:
        ll = np.nan

    print("\n=== 최종 테스트 성능 ===")
    print(f"accuracy : {acc:.4f}")
    if not np.isnan(ll):
        print(f"log_loss : {ll:.4f}")
    print("\n=== 분류 리포트 ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== 혼동행렬 ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_est, BEST_MODEL_PATH)
    print(f"\n[완료] 최적 모델 저장: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main()
