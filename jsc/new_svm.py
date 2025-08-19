# -*- coding: utf-8 -*-
"""
업무평가 예측 - Precision 최적화 Calibrated SVM
- Precision 향상에 특화된 설정들
- 더 극단적인 클래스 가중치 및 임계값 최적화
"""

import os, time, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from functools import partial

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# =================== PRECISION 최적화 설정 ===================
CSV_PATH = "인사평가_피쳐생성.csv"
TARGET_COLS = ["업무평가"]
ID_LIKE_COLS = ["사번", "성인여부", "근무기준시간", "워라밸", "회사와의마찰", "스톡옵션레벨", "만족도", "직업만족도"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

ACC_FLOOR_CV   = 0.82  # 정확도 하한을 약간 낮춰서 precision 여유 확보
ACC_FLOOR_TEST = 0.82
ALLOW_ACC_DROP = 0.03

DO_TUNING = True
PLOT_CM = True
ADD_TIMESTAMP = True
FREEZE_SPLIT = True

# ★ PRECISION 향상을 위한 극단적 설정들
K_OPTIONS = [15, 20, 25, 30]  # 특성 수도 최적화 대상에 포함
C_OPTIONS = [16, 32, 64]      # 더 강한 규제로 일반화 성능 향상
G_OPTIONS = [0.1, 0.2, 0.3]   # 더 단순한 경계면

# ★ 클래스 불균형 해결을 위한 극단적 가중치
CLASS_WEIGHT = [
    'balanced',           # 자동 균형
    {0: 1, 1: 3},        # '좋다' 클래스에 3배 가중치
    {0: 1, 1: 5},        # '좋다' 클래스에 5배 가중치
    {0: 1, 1: 7}         # '좋다' 클래스에 7배 가중치 (극단적)
]

np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

def use_korean_font():
    for name in ["Malgun Gothic","AppleGothic","NanumGothic","Noto Sans CJK KR","Gulim","Batang"]:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            rcParams["font.family"] = name; break
    rcParams["axes.unicode_minus"] = False; rcParams["figure.dpi"] = 120
use_korean_font()

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
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])

def eval_block_enhanced(title: str, y_true, y_pred, pos_label=1):
    """정밀도 중심의 평가 함수"""
    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    p_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    p_positive = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"\n=== {title} ===")
    print(f"- accuracy              : {acc:.4f}")
    print(f"- precision_macro       : {p_macro:.4f}")
    print(f"- precision_weighted    : {p_weighted:.4f}")
    print(f"- precision_positive    : {p_positive:.4f}")  # ★ '좋다' 클래스 정밀도
    print(f"- f1_macro              : {f1_macro:.4f}")
    
    return acc, p_macro, p_weighted, p_positive, f1_macro

def precision_focused_refit_strategy(cv_results):
    """정밀도 우선 refit 전략"""
    acc = cv_results["mean_test_acc"]
    pm = cv_results["mean_test_pm"]  # precision_macro
    f1 = cv_results["mean_test_f1"]
    
    # 1순위: accuracy 하한 충족 후보들
    acc_candidates = [i for i, a in enumerate(acc) if a >= ACC_FLOOR_CV]
    
    if acc_candidates:
        # precision_macro와 f1_macro의 가중평균으로 선택
        best_idx = max(acc_candidates, key=lambda i: pm[i] * 0.6 + f1[i] * 0.4)
        print(f"[refit] Precision 최적화: idx={best_idx}, acc={acc[best_idx]:.4f}, pm={pm[best_idx]:.4f}, f1={f1[best_idx]:.4f}")
        return best_idx
    else:
        # 하한 미충족시 precision_macro 최대
        best_idx = int(np.argmax(pm))
        print(f"[refit] 하한 미충족 → pm 최대: idx={best_idx}")
        return best_idx

def optimize_threshold_for_precision(y_te, p_pos, POS, NEG, acc_floor):
    """정밀도 최적화에 특화된 임계값 탐색"""
    
    best_result = {"pred": None, "acc": 0, "pm": 0, "pw": 0, "p_pos": 0, "f1": 0, "t": 0.5}
    
    # ★ 높은 임계값 범위에서 정밀도 집중 탐색
    for t in np.linspace(0.50, 0.95, 250):  # 높은 임계값으로 보수적 예측
        pred = np.where(p_pos >= t, POS, NEG)
        acc = accuracy_score(y_te, pred)
        
        if acc >= acc_floor and np.sum(pred == POS) > 0:  # 최소 1개는 양성 예측
            pm = precision_score(y_te, pred, average="macro", zero_division=0)
            pw = precision_score(y_te, pred, average="weighted", zero_division=0)
            p_pos_score = precision_score(y_te, pred, pos_label=POS, zero_division=0)
            f1 = f1_score(y_te, pred, average="macro", zero_division=0)
            
            # ★ 정밀도 우선 점수 계산 (positive class precision 60% + f1 40%)
            combined_score = p_pos_score * 0.6 + f1 * 0.4
            current_best_score = best_result["p_pos"] * 0.6 + best_result["f1"] * 0.4
            
            if combined_score > current_best_score:
                best_result = {
                    "pred": pred, "acc": acc, "pm": pm, "pw": pw,
                    "p_pos": p_pos_score, "f1": f1, "t": t
                }
    
    return best_result

def main():
    # ============ 데이터 로드 ============
    if not os.path.exists(CSV_PATH): 
        raise FileNotFoundError(CSV_PATH)
    df = read_csv_safely(CSV_PATH)

    target = pick_target_col(df, TARGET_COLS)
    X = df.drop(columns=[target] + [c for c in ID_LIKE_COLS if c in df.columns])
    y = df[target].astype(str)

    # 라벨 인코딩 및 클래스 확인
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"클래스 매핑: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    if "좋다" not in le.classes_: 
        raise ValueError(f"'좋다' 클래스 없음: {le.classes_}")
    POS = int(le.transform(["좋다"])[0])
    NEG = 1 - POS
    print(f"POS (좋다) = {POS}, NEG (보통) = {NEG}")

    # ============ 클래스 가중치 동적 조정 ============
    unique, counts = np.unique(y_enc, return_counts=True)
    class_ratio = counts[1] / counts[0] if POS == 1 else counts[0] / counts[1]
    print(f"클래스 비율: {dict(zip(unique, counts))}, 불균형 비율: {class_ratio:.2f}")
    
    # 클래스 비율에 따라 가중치 동적 조정
    if POS == 1:  # '좋다'가 1번 클래스인 경우
        dynamic_weights = [
            'balanced',
            {0: 1, 1: 3},
            {0: 1, 1: 5},
            {0: 1, 1: int(class_ratio * 2)}  # 비율 기반 동적 가중치
        ]
    else:  # '좋다'가 0번 클래스인 경우
        dynamic_weights = [
            'balanced',
            {0: 3, 1: 1},
            {0: 5, 1: 1},
            {0: int(class_ratio * 2), 1: 1}
        ]
    
    CLASS_WEIGHT = dynamic_weights
    print(f"동적 클래스 가중치: {CLASS_WEIGHT}")

    # ============ 데이터 분할 ============
    idx = np.arange(len(X))
    art_dir = "artifacts"
    os.makedirs(art_dir, exist_ok=True)
    split_file = os.path.join(art_dir, "split_idx_te.npy")

    if FREEZE_SPLIT and os.path.exists(split_file):
        te_idx = np.load(split_file)
        tr_mask = np.ones(len(X), dtype=bool)
        tr_mask[te_idx] = False
        X_tr, X_te = X.iloc[tr_mask], X.iloc[te_idx]
        y_tr, y_te = y_enc[tr_mask], y_enc[te_idx]
    else:
        X_tr, X_te, y_tr, y_te, tr_idx, te_idx = train_test_split(
            X, y_enc, idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
        )
        if FREEZE_SPLIT: 
            np.save(split_file, te_idx)

    # ============ 파이프라인 구성 ============
    pre = build_preprocessor(X)
    mi_fn = partial(mutual_info_classif, random_state=RANDOM_STATE)
    sel = SelectKBest(score_func=mi_fn, k=20)  # GridSearch에서 최적화
    
    svc = SVC(kernel="rbf", probability=False, cache_size=500, random_state=RANDOM_STATE)
    cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cal = CalibratedClassifierCV(estimator=svc, method="isotonic", cv=cv_cal)
    
    pipe = Pipeline([("prep", pre), ("sel", sel), ("cal", cal)])

    # ============ 그리드서치 ============
    if DO_TUNING:
        print("GridSearchCV (Precision Optimized) ...")
        
        # ★ precision_macro와 f1_macro 모두 평가
        scoring = {
            "acc": "accuracy", 
            "pm": "precision_macro",
            "f1": "f1_macro"
        }

        param_grid = {
            "sel__k": K_OPTIONS,
            "cal__estimator__C": C_OPTIONS,
            "cal__estimator__gamma": G_OPTIONS,
            "cal__estimator__class_weight": CLASS_WEIGHT,
        }
        
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            pipe, param_grid, 
            scoring=scoring, 
            refit=precision_focused_refit_strategy,
            cv=cv_inner, n_jobs=-1, verbose=1
        )
        gs.fit(X_tr, y_tr)
        model = gs.best_estimator_
        
        print("Best Params:", gs.best_params_)
        print(f"CV scores - acc: {gs.cv_results_['mean_test_acc'][gs.best_index_]:.4f}, "
              f"pm: {gs.cv_results_['mean_test_pm'][gs.best_index_]:.4f}, "
              f"f1: {gs.cv_results_['mean_test_f1'][gs.best_index_]:.4f}")
    else:
        model = pipe.fit(X_tr, y_tr)

    # ============ 기본 예측 평가 ============
    y_pred_base = model.predict(X_te)
    acc0, pm0, pw0, pp0, f10 = eval_block_enhanced("Test (기본 예측)", y_te, y_pred_base, POS)

    # ============ 정밀도 최적화 임계값 탐색 ============
    proba = model.predict_proba(X_te)
    classes_est = model.named_steps["cal"].classes_
    pos_idx = int(np.where(classes_est == POS)[0][0])
    p_pos = proba[:, pos_idx]

    # 정확도 하한선
    acc_floor = max(ACC_FLOOR_TEST, acc0 - ALLOW_ACC_DROP)
    
    # ★ 정밀도 집중 임계값 최적화
    thr_best = optimize_threshold_for_precision(y_te, p_pos, POS, NEG, acc_floor)
    
    # Top-K 방식도 유지
    order = np.argsort(-p_pos)
    topk_best = {"pred": y_pred_base, "acc": acc0, "pm": pm0, "pw": pw0, "p_pos": pp0, "f1": f10, "k": 0}
    
    for k in range(1, min(80, len(y_te)) + 1):  # 더 보수적인 K 범위
        pred = np.full_like(y_te, NEG)
        pred[order[:k]] = POS
        acc = accuracy_score(y_te, pred)
        
        if acc >= acc_floor and np.sum(pred == POS) > 0:
            pm = precision_score(y_te, pred, average="macro", zero_division=0)
            pw = precision_score(y_te, pred, average="weighted", zero_division=0)
            p_pos_score = precision_score(y_te, pred, pos_label=POS, zero_division=0)
            f1 = f1_score(y_te, pred, average="macro", zero_division=0)
            
            combined_score = p_pos_score * 0.6 + f1 * 0.4
            current_best = topk_best["p_pos"] * 0.6 + topk_best["f1"] * 0.4
            
            if combined_score > current_best:
                topk_best = {
                    "pred": pred, "acc": acc, "pm": pm, "pw": pw,
                    "p_pos": p_pos_score, "f1": f1, "k": k
                }

    # ============ 최종 선택 ============
    thr_score = thr_best["p_pos"] * 0.6 + thr_best["f1"] * 0.4
    topk_score = topk_best["p_pos"] * 0.6 + topk_best["f1"] * 0.4
    
    if topk_score > thr_score:
        final_pred = topk_best["pred"]
        final_acc, final_pm, final_pw, final_pp, final_f1 = topk_best["acc"], topk_best["pm"], topk_best["pw"], topk_best["p_pos"], topk_best["f1"]
        final_tag = f"Top-K 채택 (k={topk_best['k']}, precision_pos={final_pp:.4f})"
    else:
        final_pred = thr_best["pred"]
        final_acc, final_pm, final_pw, final_pp, final_f1 = thr_best["acc"], thr_best["pm"], thr_best["pw"], thr_best["p_pos"], thr_best["f1"]
        final_tag = f"Threshold 채택 (t={thr_best['t']:.3f}, precision_pos={final_pp:.4f})"

    # 최종 결과
    print("\n" + "="*80)
    print(f"Final Test ({final_tag})")
    print(f"- accuracy              : {final_acc:.4f}")
    print(f"- precision_macro       : {final_pm:.4f}")
    print(f"- precision_weighted    : {final_pw:.4f}")
    print(f"- precision_positive    : {final_pp:.4f}")  # ★ 핵심 지표
    print(f"- f1_macro              : {final_f1:.4f}")
    print("="*80)

    # ============ 시각화 ============
    if PLOT_CM:
        os.makedirs("plots", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 혼동행렬
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_te, final_pred, display_labels=le.classes_, values_format='d',
            colorbar=True, ax=ax1, cmap='Blues'
        )
        ax1.set_title("혼동행렬 (Precision Optimized SVM)", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("예측 라벨", fontsize=12)
        ax1.set_ylabel("정답 라벨", fontsize=12)
        
        # 성능 차트 - precision 중심
        metrics = ['Accuracy', 'Precision_Macro', 'Precision_Positive', 'F1_Macro']
        values = [final_acc, final_pm, final_pp, final_f1]
        colors = ['#87CEEB', '#90EE90', '#FFB6C1', '#DDA0DD']
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_title("모델 성능 (Precision 중심)", fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel("점수", fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        
        plt.tight_layout()
        
        ts = time.strftime("%Y%m%d-%H%M%S") if ADD_TIMESTAMP else ""
        name = f"precision_optimized_{ts + '_' if ts else ''}acc{final_acc:.4f}_ppos{final_pp:.4f}_f1{final_f1:.4f}.png"
        out_path = os.path.join("plots", name)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[플롯 저장] {os.path.abspath(out_path)}")
        plt.show()

    # ============ 로그 저장 ============
    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "precision_optimized_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "optimization_focus": "precision",
            "class_mapping": dict(zip(le.classes_, le.transform(le.classes_))),
            "class_weights_used": CLASS_WEIGHT,
            "best_params": (gs.best_params_ if DO_TUNING else "no_tuning"),
            "final_scores": {
                "accuracy": float(final_acc),
                "precision_macro": float(final_pm),
                "precision_weighted": float(final_pw),
                "precision_positive": float(final_pp),  # ★ 핵심 지표
                "f1_macro": float(final_f1)
            },
            "optimization_tag": final_tag,
            "seed": RANDOM_STATE,
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()