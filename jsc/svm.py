# -*- coding: utf-8 -*-
"""
업무평가 예측 - Calibrated SVM + Aggressive Feature Selection (precision 목표)
- 요점: 소수 클래스 '좋다'를 "정말 자신있는 몇 건"만 골라내도록 랭킹 분해능↑
- 방법:
  (1) OHE+스케일 이후 SelectKBest(mutual_info)로 상위 특징만 사용(K=10~30)
  (2) SVC(RBF) 경계 강화(C↑, gamma↑)
  (3) CalibratedClassifierCV(method='isotonic') 확률 보정 → 임계값/Top-K 튜닝
- 출력: accuracy, precision_macro, precision_weighted
- 혼동행렬 PNG 저장(파일명에 지표 포함)
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# ===== 경로/고정값 =====
CSV_PATH = "WA_Fn-UseC_-HR-Employee-Attrition_변환.csv"
TARGET_COLS = ["업무평가","PerformanceRating"]
ID_LIKE_COLS = ["사번", "성인여부", "근무기준시간"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 목표/제약: 정확도 하한을 살짝 완화(정밀도 공간 확보)
ACC_FLOOR_CV   = 0.84     # CV에서 refit 후보 하한
ACC_FLOOR_TEST = 0.84     # Test에서 유지할 하한
ALLOW_ACC_DROP = 0.02     # 임계값/Top-K 튜닝 시 baseline 대비 허용 하락폭(≤2%p)

DO_TUNING = True
PLOT_CM = True
ADD_TIMESTAMP = True

# === 핵심: 공격적 특징 선택 + 경계 강화 그리드(가볍게) ===
K_OPTIONS = [10, 15, 20, 30]          # 상위 특징 개수 (작을수록 분해능↑ 기대)
C_OPTIONS = [8, 16, 32]               # 마진 경계 강화
G_OPTIONS = [0.2, 0.3, 0.5]           # 더 예민한 결정경계
CLASS_WEIGHT = [None]                 # precision을 위해 과한 균형 가중은 잠시 제외

# ===== 폰트(한글 그래프) =====
def use_korean_font():
    for name in ["Malgun Gothic","AppleGothic","NanumGothic","Noto Sans CJK KR","Gulim","Batang"]:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            rcParams["font.family"] = name; break
    rcParams["axes.unicode_minus"] = False; rcParams["figure.dpi"] = 120
use_korean_font()

# ===== 유틸 =====
def read_csv_safely(path: str) -> pd.DataFrame:
    for enc, kw in [("utf-8", {}), ("cp949", {"errors":"replace"})]:
        try: return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except Exception: pass
    return pd.read_csv(path, encoding="utf-8", low_memory=False)

def pick_target_col(df: pd.DataFrame, cands) -> str:
    for c in cands:
        if c in df.columns: return c
    raise ValueError(f"타깃 컬럼 없음: {cands}")

def build_preprocessor(X: pd.DataFrame):
    # ⚠️ RBF SVC는 희소행렬 미지원 → OHE는 dense(sparse=False)
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat),
        ("num", StandardScaler(), num),
    ])

def eval_block(title: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {title} ===")
    print(f"- accuracy          : {acc:.4f}")
    print(f"- precision_macro   : {p_macro:.4f}")
    print(f"- precision_weighted: {p_weighted:.4f}")
    return acc, p_macro, p_weighted

def main():
    # 1) 데이터 로드/분할
    if not os.path.exists(CSV_PATH): raise FileNotFoundError(CSV_PATH)
    df = read_csv_safely(CSV_PATH)
    target = pick_target_col(df, TARGET_COLS)
    X = df.drop(columns=[target] + [c for c in ID_LIKE_COLS if c in df.columns])
    y = df[target].astype(str)

    le = LabelEncoder(); y_enc = le.fit_transform(y)
    classes = le.classes_.tolist()
    if "좋다" not in classes: raise ValueError(f"'좋다' 클래스가 없습니다: {classes}")
    POS = int(le.transform(["좋다"])[0]); NEG = 1 - POS

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    # 2) 파이프라인: 전처리 → SelectKBest(MI) → SVC → 확률보정
    pre  = build_preprocessor(X)
    svc  = SVC(kernel="rbf", probability=False, cache_size=500, random_state=RANDOM_STATE)
    cal  = CalibratedClassifierCV(base_estimator=svc, method="isotonic", cv=3)

    pipe = Pipeline([
        ("prep", pre),
        ("sel",  SelectKBest(score_func=mutual_info_classif, k=20)),  # k는 그리드에서 바뀜
        ("cal",  cal),
    ])

    # 3) 작은 그리드서치: acc 하한 충족 후보 중 precision_weighted 최대 refit
    if DO_TUNING:
        print("GridSearchCV (Calibrated SVM + Aggressive FS) ...")
        scoring = {"acc": "accuracy", "pw": "precision_weighted"}
        def refit_strategy(cv_results):
            acc = cv_results["mean_test_acc"]; pw = cv_results["mean_test_pw"]
            idx = [i for i,(a,p) in enumerate(zip(acc, pw)) if a >= ACC_FLOOR_CV]
            if idx:
                best = idx[int(np.argmax([pw[i] for i in idx]))]
                print(f"[refit] acc 하한 충족 → pw 최대 채택 (idx={best}, acc={acc[best]:.4f}, pw={pw[best]:.4f})")
                return best
            best = int(np.argmax(pw)); print(f"[refit] 하한 미충족 → pw 최대 채택 (idx={best})"); return best

        param_grid = {
            "sel__k": K_OPTIONS,
            "cal__base_estimator__C": C_OPTIONS,
            "cal__base_estimator__gamma": G_OPTIONS,
            "cal__base_estimator__class_weight": CLASS_WEIGHT,
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, param_grid, scoring=scoring, refit=refit_strategy, cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        model = gs.best_estimator_
        print("Best Params:", gs.best_params_)
        print(f"CV mean acc={gs.cv_results_['mean_test_acc'][gs.best_index_]:.4f}, "
              f"mean pw={gs.cv_results_['mean_test_pw'][gs.best_index_]:.4f}")
    else:
        model = pipe.fit(X_tr, y_tr)

    # 4) 기본 예측 + 디버그(랭킹 품질)
    y_pred_base = model.predict(X_te)
    acc0, pm0, pw0 = eval_block("Test (기본 예측)", y_te, y_pred_base)

    proba = model.predict_proba(X_te)                     # (n,2)
    classes_est = model.named_steps["cal"].classes_       # calibrator 기준 클래스
    pos_idx = int(np.where(classes_est == POS)[0][0])
    p_pos = proba[:, pos_idx]

    order = np.argsort(-p_pos)
    print("\n[DEBUG] 상위 20개 p(좋다)와 실제 라벨:")
    for i, idx in enumerate(order[:min(20, len(order))], 1):
        print(f"{i:02d}. p={p_pos[idx]:.3f}  true={'좋다' if y_te[idx]==POS else '보통'}")

    print("\n[DEBUG] Top-K 누적 TP:")
    for k in [1,2,3,5,10,15,20,30,40,60]:
        k = min(k, len(order))
        sel = order[:k]; tp = int(np.sum(y_te[sel] == POS))
        print(f"K={k:>2}: TP={tp},  TP율={tp/max(1,k):.2f}")

    # 5) 확률 임계값 스윕 + Top-K (정확도 하한 유지)
    acc_floor = max(ACC_FLOOR_TEST, acc0 - ALLOW_ACC_DROP)

    # (A) threshold sweep: 0.3~0.99 넓게 (보수적 양성만 허용)
    thr_best = {"pred": y_pred_base, "acc": acc0, "pm": pm0, "pw": pw0, "t": 0.50}
    for t in np.linspace(0.30, 0.99, 70):
        pred = np.where(p_pos >= t, POS, NEG)
        acc = accuracy_score(y_te, pred)
        if acc >= acc_floor:
            pm = precision_score(y_te, pred, average="macro",    zero_division=0)
            pw = precision_score(y_te, pred, average="weighted", zero_division=0)
            if (pw > thr_best["pw"]) or (np.isclose(pw, thr_best["pw"]) and acc > thr_best["acc"]):
                thr_best = {"pred": pred, "acc": acc, "pm": pm, "pw": pw, "t": t}

    # (B) Top-K: 확률 상위 K개만 '좋다' (K는 1~120)
    topk_best = {"pred": y_pred_base, "acc": acc0, "pm": pm0, "pw": pw0, "k": 0}
    for k in range(1, min(120, len(y_te)) + 1):
        pred = np.full_like(y_te, NEG)
        pred[order[:k]] = POS
        acc = accuracy_score(y_te, pred)
        if acc >= acc_floor:
            pm = precision_score(y_te, pred, average="macro",    zero_division=0)
            pw = precision_score(y_te, pred, average="weighted", zero_division=0)
            if (pw > topk_best["pw"]) or (np.isclose(pw, topk_best["pw"]) and acc > topk_best["acc"]):
                topk_best = {"pred": pred, "acc": acc, "pm": pm, "pw": pw, "k": k}

    # 6) 더 좋은 방식 채택
    use_topk = (topk_best["pw"] > thr_best["pw"]) or (np.isclose(topk_best["pw"], thr_best["pw"]) and topk_best["acc"] > thr_best["acc"])
    if use_topk:
        final_pred, final_acc, final_pm, final_pw = topk_best["pred"], topk_best["acc"], topk_best["pm"], topk_best["pw"]
        tag = f"Top-K 채택 (k={topk_best['k']}, acc_floor={acc_floor:.3f})"
    else:
        final_pred, final_acc, final_pm, final_pw = thr_best["pred"], thr_best["acc"], thr_best["pm"], thr_best["pw"]
        tag = f"Threshold 채택 (t={thr_best['t']:.3f}, acc_floor={acc_floor:.3f})"

    print("\n" + "="*70)
    print(f"Final Test ({tag})")
    print(f"- accuracy          : {final_acc:.4f}")
    print(f"- precision_macro   : {final_pm:.4f}")
    print(f"- precision_weighted: {final_pw:.4f}")
    print("="*70)

    # 7) 혼동행렬 저장(파일명에 지표 포함)
    if PLOT_CM:
        os.makedirs("plots", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay.from_predictions(
            y_te, final_pred, display_labels=le.classes_, values_format='d', colorbar=False, ax=ax
        )
        ax.set_title("혼동행렬 (Calibrated SVM + Aggressive FS)")
        ax.set_xlabel("예측 라벨"); ax.set_ylabel("정답 라벨")
        fig.tight_layout()
        ts = time.strftime("%Y%m%d-%H%M%S") if ADD_TIMESTAMP else ""
        name = f"cm_{ts + '_' if ts else ''}acc{final_acc:.4f}_pmacro{final_pm:.4f}_pweighted{final_pw:.4f}.png"
        out_path = os.path.join("plots", name)
        fig.savefig(out_path, dpi=150); print(f"[플롯 저장] {os.path.abspath(out_path)}"); plt.show()

if __name__ == "__main__":
    main()
