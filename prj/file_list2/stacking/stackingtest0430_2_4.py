import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
import catboost as cb
from catboost import CatBoostRegressor
import torch
import os
import shap
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
import sys
import lightgbm
import datetime
import time
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from catboost import CatBoostClassifier, Pool
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import learning_curve
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# 必要に応じて評価（回帰指標で）
from sklearn.metrics import mean_squared_error, r2_score
#https://rin-effort.com/2020/02/09/learn-stacking/#toc1
start = time.time()

#処理時間出力

# 現在のスクリプトのパスを取得
current_script_path = os.path.abspath(__file__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#出力用フォルダ
dt_now = datetime.datetime.now()
output_folder_time = dt_now.microsecond
test_idx = 9

output_folder = f"output_images/0127_senbatu/stacking0430_2_4/0704/test{test_idx}/10"
os.makedirs(output_folder, exist_ok=True)

#cnn/select_dataset.pyにより，男女の分布が均等になるようにデータセットを作成
#それらのデータセットをリストにまとめる
file_paths = [f"../csv/0127_2/random_data{i}.csv" for i in range(10)]#rangeの変更を忘れない！

#各CSVファイルを読み込み
def load_file(file_path):
    return pd.read_csv(file_path)
# ファイルをロード
target = 'chip_mean'
X_test = load_file(file_paths[test_idx])

#X_val = load_file(file_paths[val_idx])
train_files = [
    file_paths[i] for i in range(len(file_paths)) if i != test_idx# and  i != val_idx
]
X_train = pd.concat([load_file(fp) for fp in train_files], axis=0)
print(train_files)
print("Unique rows in training data:", X_train.drop_duplicates().shape[0])

#説明変数と目的変数に分割
"""
#ハズレ値除去
y_train_ori = X_train[target]
first_mask = y_train_ori < 2.5
X_train = X_train[first_mask]
y_train = y_train_ori[first_mask]
y_train_fil = y_train
y_train = y_train.apply(lambda val: 0 if val < 0.5 else 1)
X_train = X_train.drop(columns = [target])
"""
y_train_ori = X_train[target]
y_train = y_train_ori.copy()
y_train_fil = y_train
y_train = y_train.apply(lambda val: 0 if val < 0.5 else 1)
X_train = X_train.drop(columns = [target])
#X_train = X_train.drop(columns = ['Index','angle1','Proximal1','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])
#X_train = X_train.drop(columns = ['Index','angle2', 'angle3','Proximal2','Proximal3','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])
X_train = X_train.drop(columns = ['Index','limitation','restored','remaining','gender','body','Nofc','width','placement1','placement2','bone','edentulous','angle2', 'angle3','Proximal2','Proximal3','quality','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])
#'placement1','placement2',

"""#ハズレ値除去
first_mask = y_test_ori < 2.5
X_test = X_test[first_mask]
y_test = y_test_ori[first_mask]
y_test_fil = y_test
y_test = y_test.apply(lambda val: 0 if val < 0.5 else 1)
X_test = X_test.drop(columns = [target])
"""
y_test_ori = X_test[target]
y_test = y_test_ori.copy()
y_test_fil = y_test
y_test = y_test.apply(lambda val: 0 if val < 0.5 else 1)
X_test = X_test.drop(columns = [target])
#X_test = X_test.drop(columns = ['Index','angle1','Proximal1','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])
#X_test = X_test.drop(columns = ['Index','angle2', 'angle3','Proximal2','Proximal3','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])
X_test = X_test.drop(columns = ['Index','limitation','restored','remaining','gender','body','Nofc','width','placement1','placement2','bone','edentulous','angle2', 'angle3','Proximal2','Proximal3','quality','entry_mino','entry_kurosaki','entry_mean','chip_mino','chip_kurosaki'])

#正規化
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#保存用
output_tree = os.path.join(output_folder, "tree")
os.makedirs(output_tree, exist_ok=True)
output_density = os.path.join(output_folder, "density")
os.makedirs(output_density, exist_ok=True)
output_bar = os.path.join(output_folder, "bar")
os.makedirs(output_bar, exist_ok=True)
output_waterfall = os.path.join(output_folder, "waterfall")
os.makedirs(output_waterfall, exist_ok=True)
output_heatmap = os.path.join(output_folder, "kobetu_heatmap")
os.makedirs(output_heatmap, exist_ok=True)
output_roc = os.path.join(output_folder, "roc")
os.makedirs(output_roc, exist_ok=True)
output_results = os.path.join(output_folder, "results")
os.makedirs(output_results, exist_ok=True)
output_boxplot = os.path.join(output_folder, "boxplot")
os.makedirs(output_boxplot, exist_ok=True)
output_plot = os.path.join(output_folder, "plot")
os.makedirs(output_plot, exist_ok=True)

def shaptry(model, X_val, id):
    #shapを試す
    explainer = shap.TreeExplainer(model=model)
    print(explainer.expected_value)
    X_test_shap = X_val.copy().reset_index(drop=True)
    shap_values = explainer.shap_values(X=X_test_shap)
    
    #密度プロットの保存
    shap.summary_plot(shap_values, X_test_shap, show=False) #左側の図
    plt.savefig(os.path.join(output_density, f"plot_density_{id}.png"), bbox_inches="tight")
    plt.clf()
    #バーの保存
    shap.summary_plot(shap_values, X_test_shap, plot_type='bar', show=False) #右側の図
    plt.savefig(os.path.join(output_bar, f"plot_bar_{id}.png"), bbox_inches="tight")
    plt.clf()

    n = 0#テストデータのn番目の要素を指定
    force_plot = shap.force_plot(explainer.expected_value, shap_values[n, :], X_test_shap.iloc[n, :])
    shap.save_html(os.path.join(output_density, f"force_plot_{id}.html"), force_plot)  # HTML保存

    #waterfall_plotは私の環境ではエラーになるので、代わりにwaterfall_legacyを使用している。
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, 
                                        shap_values[n,:], X_test_shap.iloc[n,:],show=False) #下の図
    plt.savefig(os.path.join(output_waterfall, f"plot_waterfall_{id}.png"), bbox_inches="tight")
    plt.clf()

#モデルの定義

def train_lgb(X, y, X_val=None, y_val=None, X_test=None):
    params={
        'objective': 'binary',
        'metric': 'binary_logloss',
        'task':'train',
        'boosting_type':'gbdt',
        'max_depth': 7, 
        #'eta': 0.23876525248833627, 
        'eta': 0.08, 
        'lambda_l1': 0.5, 
        'lambda_l2': 0.5,
        'seed':42   
    }

    lgb_train = lgb.Dataset(X, y)
    evaluation_results = {}

    if X_val is not None and y_val is not None:
        lgb_eval = lgb.Dataset(X_val, y_val)
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_names=['train', 'valid'],
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=True),
                lgb.record_evaluation(evaluation_results)
            ]
        )

        # 学習曲線のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(evaluation_results['train']['binary_logloss'], label='Train Loss')
        plt.plot(evaluation_results['valid']['binary_logloss'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Binary Log Loss')
        plt.title('学習曲線 (LightGBM)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, "learning_curve_lgb1.png"))
        plt.close()

        # SHAP 可視化
        id = 4
        shaptry(model, X_val, id)

        return model.predict(X_val)

    elif X_test is not None:
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000
        )

        return model.predict(X_test)

    else:
        raise ValueError("Either X_val and y_val or X_test must be provided.")


def train_xgb(X, y, X_val=None,y_val=None,X_test=None):
    #params = {'objective': 'binary:logistic', 'eval_metric': 'logloss','max_depth':2,'eta':0.2, 'early_stopping_rounds':10,'random_state':42}
    if X_val is not None and y_val is not None:
        params = {
        'objective': 'binary:logistic',      # 目的関数：多値分類、マルチクラス分類
        'eval_metric': 'logloss',      # 分類モデルの性能を測る指標
        'learning_rate':0.1 ,          # 学習率（初期値0.1）
        'max_depth':7,
        'eta':0.06,
        'lambda': 0.5,
        'alpha': 0.5,
        'early_stopping_rounds':10,
        'random_state':42
        }
        model = xgb.XGBClassifier(**params, n_estimators=100)
        evals_result = {}
        model.fit(X, y, eval_set=[(X,y),(X_val, y_val)], verbose=True)
        id = 1
        evals_result = model.evals_result()
        shaptry(model, X_val, id)
        # 学習曲線のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result['validation_0']['logloss'], label='Train Loss')
        plt.plot(evals_result['validation_1']['logloss'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('学習曲線 (XGBoost)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"learning_curve_xgb1.png"))
        plt.close()
        return model.predict_proba(X_val), model.predict(X_val)
    elif X_test is not None:
        params = {
        'objective': 'binary:logistic',      # 目的関数：多値分類、マルチクラス分類
        'eval_metric': 'logloss',      # 分類モデルの性能を測る指標
        'max_depth':7,
        'eta':0.07,
        'lambda': 0.5,
        'alpha': 0.5,
        'random_state':42
        }
        model = xgb.XGBClassifier(**params, n_estimators=100)
        model.fit(X, y, verbose=True)
        return model.predict_proba(X_test), model.predict(X_test)
    else:
        raise ValueError("Either X_val and y_val or X_test must be provided.")
            
        
def train_xgb2(X, y, X_val=None,y_val=None,X_test=None):
    #params = {'objective': 'binary:logistic', 'eval_metric': 'logloss','max_depth':2,'eta':0.2, 'early_stopping_rounds':10,'random_state':42}
    if X_val is not None and y_val is not None:
        params = {
        'objective': 'binary:logistic',      # 目的関数：多値分類、マルチクラス分類
        'eval_metric': 'logloss',      # 分類モデルの性能を測る指標
        'learning_rate':0.1 ,          # 学習率（初期値0.1）
        'max_depth':7,
        'eta':0.059253000761891446,
        'subsample':0.9152831131977142,
        'colsample_bytree': 0.6501429315627342,
        'lambda': 0.6638411782255692,
        'alpha': 0.3091607516357824,
        'early_stopping_rounds':10,
        'random_state':42
        }
        model = xgb.XGBClassifier(**params, n_estimators=100)
        evals_result = {}
        model.fit(X, y, eval_set=[(X,y),(X_val, y_val)], verbose=True)
        id = 1
        evals_result = model.evals_result()
        
        shaptry(model, X_val, id)
        # 学習曲線のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result['validation_0']['logloss'], label='Train Loss')
        plt.plot(evals_result['validation_1']['logloss'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('学習曲線 (XGBoost)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"learning_curve_xgb1.png"))
        plt.close()
        return model.predict_proba(X_val), model.predict(X_val)
    elif X_test is not None:
        params = {
        'objective': 'binary:logistic',      # 目的関数：多値分類、マルチクラス分類
        'eval_metric': 'logloss',      # 分類モデルの性能を測る指標
        'max_depth':3,
        'eta':0.1,
        'subsample':0.9152831131977142,
        'colsample_bytree': 0.6501429315627342,
        'lambda': 0.5,
        'alpha': 0.5,
        'random_state':42
        }
        model = xgb.XGBClassifier(**params, n_estimators=100)
        model.fit(X, y, verbose=True)
        return model.predict_proba(X_test), model.predict(X_test)
    else:
        raise ValueError("Either X_val and y_val or X_test must be provided.")

def train_cb(X, y,  X_val=None, y_val=None, X_test=None):
    params = {
    'iterations': 405,
    'depth': 4,
    'od_type': 'IncToDec',
    'loss_function': 'Logloss',
    'od_wait':30,
    'random_state':42
    }
    #訓練用
    train_pool = Pool(data=X, label = y)
    if X_val is not None and y_val is not None:
        #検証用
        val_pool = Pool(data=X_val, label=y_val)
        # 学習
        model = CatBoostClassifier(**params)  
        model.fit(train_pool, eval_set=val_pool,early_stopping_rounds=params['od_wait'])       
        id = 2
        shaptry(model, X_val, id)
            #学習曲線
        history = model.get_evals_result()
        print(history)
        # グラフにプロットする
        train_metric = history['learn']['Logloss']
        eval_metric = history['validation']['Logloss']
        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label=f'train metric (trial {test_idx+1})')
        plt.plot(eval_metric, label=f'eval metric (trial {test_idx+1})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"learning_curve_cat1.png"))
        plt.close()
        return model.predict_proba(X_val),model.predict(X_val)

    elif X_test is not None:
        model = CatBoostClassifier(**params)  
        model.fit(X, y,early_stopping_rounds=params['od_wait']) 
        return model.predict_proba(X_test),model.predict(X_test)
    else:
        raise ValueError("Either X_val and y_val or X_test must be provided.")
def train_cb2(X, y, X_val=None,y_val=None,X_test=None):
    """    params = {
        'iterations': 400,
        'depth': 7,
        'od_type': 'IncToDec',
        'loss_function': 'Logloss',
        'od_wait':30,
        'random_state':1
        }"""
    params = {
        'iterations': 300,
        'depth': 8,
        'learning_rate': 0.01,
        'random_strength': 53,
        'bagging_temperature': 19.09982941,
        'od_type': 'IncToDec',
        'od_wait': 31,
        'loss_function': 'Logloss',
        'verbose': 0,
        'random_state':42
        }

    #訓練用
    train_pool = Pool(data=X, label = y)
    if X_val is not None and y_val is not None:
        #検証用
        val_pool = Pool(data=X_val, label=y_val)
        #学習
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool,early_stopping_rounds=params['od_wait'])    
        id = 6
        shaptry(model, X_val, id)
        #学習曲線
        history = model.get_evals_result()
        print(history)
        # グラフにプロットする
        train_metric = history['learn']['Logloss']
        eval_metric = history['validation']['Logloss']
        plt.figure(figsize=(8, 6))
        plt.plot(train_metric, label=f'train metric (trial {test_idx+1})')
        plt.plot(eval_metric, label=f'eval metric (trial {test_idx+1})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_folder, f"learning_curve_cat2.png"))
        plt.close()
        return model.predict_proba(X_val),model.predict(X_val)
    elif X_test is not None:
        model = CatBoostClassifier(**params)  
        model.fit(X, y,early_stopping_rounds=params['od_wait']) 
        return model.predict_proba(X_test),model.predict(X_test)
    else:
        return ValueError("Either X_val and y_val or X_test must be provided.")

def train_rf(X,y,X_test):
    #model = RandomForestClassifier(n_estimators=300, min_samples_split = 4,criterion='entropy', max_depth=3,random_state=42)
    model = RandomForestClassifier(n_estimators=50, min_samples_split=3, min_samples_leaf=5, criterion='entropy', max_depth=2,class_weight='balanced',random_state=42)
    model.fit(X, y)
    #学習曲線
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # 学習曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label="Train Score")
    plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', label="Validation Score")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve(Random Forest)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, f"learning_curve_rf.png"))
    plt.close()

    return model.predict_proba(X_test),model.predict(X_test)

def train_svm(X,y,X_test):
    model = LinearSVC()
    model.fit(X, y)
    return model._predict_proba_lr(X_test), model.predict(X_test)

def confusion(y_test, final_predict, idx):  # 個別の識別器の分類結果確認用
    confusion = confusion_matrix(y_true=y_test, y_pred=final_predict)

    # クラス 1 が 1 行目および 1 列目に来るように行列の順番を変更
    confusion = confusion[::-1, ::-1]
    
    # クラスラベルの順番を変更
    class_labels = ["Class 1", "Class 0"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.savefig(os.path.join(output_heatmap, f'heatmap_{idx}.png'))
    plt.close()

    # 各評価指標の計算
    accuracy = accuracy_score(y_test, final_predict)
    precision = precision_score(y_test, final_predict, pos_label=1)  # クラス 1 をポジティブクラスとして明示
    recall = recall_score(y_test, final_predict, pos_label=1)
    f1 = f1_score(y_test, final_predict, pos_label=1)

    # クラスラベルの順番を変更して classification_report を生成
    print(classification_report(y_test, final_predict, target_names=["Class 0", "Class 1"]))

    # 結果を result.txt として保存
    results = f"""
    ==========================
    精度評価
    ==========================
    {classification_report(y_test, final_predict, target_names=["Class 0", "Class 1"])}
    Accuracy  : {accuracy:.4f}
    Precision : {precision:.4f}
    Recall    : {recall:.4f}
    F1 Score  : {f1:.4f}
    """

    output_results_file = os.path.join(output_results, f"result{idx}.txt")
    with open(output_results_file, "w", encoding="utf-8") as f:
        f.write(results)

def best_th(y_test, meta_features_test,idx):
        # True Positive Rate, False Positive Rateの計算
    fpr, tpr, thresholds = roc_curve(y_test, meta_features_test)
    # Youden's Indexを用いて最適なしきい値を決定
    youden_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_index]
    print(f"Optimal threshold {idx}(Youden's Index): {best_threshold}")
    #ROC曲線の描画
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_roc, f"roc{idx}.png"))
    plt.close()
    return best_threshold

def best_th2(y_test, meta_features_test,idx):
        # True Positive Rate, False Positive Rateの計算
    fpr, tpr, thresholds = roc_curve(y_test, meta_features_test)
    # Youden's Indexを用いて最適なしきい値を決定
    youden_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_index]
    # 結果を result.txt として保存
    results = f"""
    ==========================
    最適なしきい値({idx})
    ==========================
    {best_threshold}
    """
    output_results_file = os.path.join(output_results, f"best_threshold{idx}.txt")
    with open(output_results_file, "w", encoding="utf-8") as f:
        f.write(results)

    return best_threshold

# クロスバリデーションでスタッキング
kf = KFold(n_splits=3, shuffle=True, random_state=42)
np.set_printoptions(suppress=True, precision=4)
list_size = 6
meta_features_train = np.zeros((X_train.shape[0], list_size-1))  # 各モデルの予測値
meta_features_train2 =  np.zeros((X_train.shape[0], 3)) 
meta_features_test = np.zeros((X_test.shape[0], list_size-1))
meta_features_test1 = np.zeros((X_test.shape[0], list_size-1))
meta_features_test2 = np.zeros((X_test.shape[0], 3))
final_train = np.zeros((X_train.shape[0], list_size))
final_test = np.zeros((X_test.shape[0], list_size))
final_train_all = np.zeros((X_train.shape[0], list_size))
final_test_all = np.zeros((X_test.shape[0], list_size))
final_result_1 = np.zeros((X_test.shape[0], list_size))
final_val = np.zeros((len(X_train), list_size))
predict_1 = np.zeros((X_test.shape[0], list_size))
diff = np.zeros((X_train.shape[0]))
diff_test = np.zeros((X_test.shape[0]))
test_diff = np.zeros((X_test.shape[0], list_size-1))
row_max = np.zeros((X_train.shape[0]))
row_min = np.zeros((X_train.shape[0]))
row_max_test = np.zeros((X_test.shape[0]))
row_min_test = np.zeros((X_test.shape[0]))
count = 0
conf_test_sum = 0
val_idx_list = []
for train_idx, val_idx in kf.split(X_train):
    #訓練データ・検証データに分割
    val_idx_list.append(val_idx) 
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    print(f"Training data: {X_tr.shape}, Validation data: {X_val.shape}")
    
    # 各モデルの予測値をメタ特徴量として収集
    modellist0 = train_lgb(X_tr, y_tr, X_val,y_val)
    #modellist1, model_proba1 = train_xgb(X_tr, y_tr, X_val,y_val)
    modellist1, model_proba1 = train_cb(X_tr, y_tr, X_val,y_val)
    modellist2, model_proba2 = train_cb2(X_tr, y_tr, X_val,y_val)
    modellist3, model_proba3 = train_svm(X_tr, y_tr, X_val)
    modellist4, model_proba4 = train_rf(X_tr, y_tr, X_val)

    #メタ特徴量・訓練データ（1である確率）
    meta_features_train[val_idx, 0] = modellist0
    #meta_features_train[val_idx, 1] = modellist1[:,1]
    meta_features_train[val_idx, 1] = modellist1[:,1]
    meta_features_train[val_idx, 2] = modellist2[:,1]
    meta_features_train[val_idx, 3] = modellist3[:,1]
    meta_features_train[val_idx, 4] = modellist4[:,1]
    sum_best_th = 0
    sum1 = 0
for idx in range(5):
    sum_best_th += best_th2(y_val, meta_features_train[val_idx,idx],idx)#各識別器ごとのしきい値を決定する（識別器の数だけしきい値が存在）→全て足し合わせる
print("sum_best_th",sum_best_th)

##################################################################################################################
#予測結果を訓練データと検証データに分ける
X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(meta_features_train, y_train, test_size=0.1,random_state=42)
meta_features_train1 =  np.zeros((X_train_meta.shape[0], list_size-1)) 
meta_features_val1 =  np.zeros((X_val_meta.shape[0], list_size-1)) 
#X_train_meta
sum_train = X_train_meta.sum(axis=1)#各行の合計値を調べる
row_max_train = np.max(X_train_meta, axis=1)
row_min_train = np.min(X_train_meta[:, :4], axis=1)
diff_train = row_max_train - row_min_train
mean_train = np.mean(X_train_meta, axis=1)
sum1_mean_train = sum_train[sum_train != 0].mean()

meta_features_train1[:,0] = sum_train
meta_features_train1[:,1] = row_max_train
meta_features_train1[:,2] = row_min_train
meta_features_train1[:,3] = mean_train

#X_val_meta
sum_val = X_val_meta.sum(axis=1)#各行の合計値を調べる
row_max_val = np.max(X_val_meta, axis=1)
row_min_val = np.min(X_val_meta[:, :4], axis=1)
diff_val = row_max_val - row_min_val
mean_val = np.mean(X_val_meta, axis=1)
sum1_mean_val = sum_val[sum_val != 0].mean()

meta_features_val1[:,0] = sum_val
meta_features_val1[:,1] = row_max_val
meta_features_val1[:,2] = row_min_val
meta_features_val1[:,3] = mean_val

feature_names = ["sum1", "std1", "max1", "min1", "mean"]

#箱ひげ図
df_plot = pd.DataFrame(meta_features_train1, columns=feature_names)
df_plot["y_train"] = y_train.reset_index(drop=True)
for col in feature_names:
    plt.figure()
    sns.boxplot(x="y_train", y=col, data=df_plot)
    plt.title(f"{col} by class")
    plt.savefig(os.path.join(output_boxplot,f"{col}_train.png"))
    plt.clf

#plot
for i in range(5):
    plt.figure()
    plt.title(f"{feature_names[i]} vs y_train")
    plt.xlabel("y_train")
    plt.ylabel(feature_names[i])
    plt.scatter(y_train_meta, meta_features_train1[:, i], alpha=0.5)
    plt.grid(True)
    plt.savefig(os.path.join(output_plot,f"{i}_train.png"))
    plt.clf

train_diff = np.zeros((X_train_meta.shape[0], list_size-1))
val_diff = np.zeros((X_val_meta.shape[0], list_size-1))
#訓練データ
for model_idx in range(list_size-1):
    thresholds_conf = best_th2(y_train_meta, X_train_meta[:, model_idx],model_idx)#列ごとにしきい値を取得
    print(f"Optimal threshold {model_idx}(Youden's Index): {thresholds_conf}")  
    for i in range(len(X_train_meta)):
        if thresholds_conf - 0.3 > X_train_meta[i, model_idx]:
            train_diff[i, model_idx] = 1
        elif thresholds_conf - 0.3 <= X_train_meta[i, model_idx] < thresholds_conf-0.2:
            train_diff[i, model_idx] = 5
        elif thresholds_conf - 0.2 <= X_train_meta[i, model_idx] < thresholds_conf-0.1:
            train_diff[i, model_idx] = 10
        elif thresholds_conf -0.1 <= X_train_meta[i, model_idx] < thresholds_conf:
            train_diff[i, model_idx] = 15
        elif thresholds_conf  <= X_train_meta[i, model_idx]:
            train_diff[i, model_idx] = 30
        else:
            train_diff[i, model_idx] = 0
#検証データ
for model_idx in range(list_size-1):
    thresholds_conf = best_th2(y_val_meta, X_val_meta[:, model_idx],model_idx)#列ごとにしきい値を取得
    print(f"Optimal threshold {model_idx}(Youden's Index): {thresholds_conf}")  
    for i in range(len(X_val_meta)):
        if thresholds_conf - 0.3 > X_val_meta[i, model_idx]:
            val_diff[i, model_idx] = 1
        elif thresholds_conf - 0.3 <= X_val_meta[i, model_idx] < thresholds_conf-0.2:
            val_diff[i, model_idx] = 5
        elif thresholds_conf - 0.2 <= X_val_meta[i, model_idx] < thresholds_conf-0.1:
            val_diff[i, model_idx] = 10
        elif thresholds_conf -0.1 <= X_val_meta[i, model_idx] < thresholds_conf:
            val_diff[i, model_idx] = 15
        elif thresholds_conf  <= X_val_meta[i, model_idx]:
            val_diff[i, model_idx] = 30
        else:
            val_diff[i, model_idx] = 0
##########################################################################################
#特徴量とラベル
X_select = meta_features_train1
X_meta = X_train_meta
X_diff = train_diff
y_f = y_train
y_meta_f = y_train_meta

print(y_f)

# k個の有用な特徴量を選択
#X_selectについて
selector = SelectKBest(score_func=f_classif, k='all')
selector_mut = SelectKBest(score_func=mutual_info_classif, k='all')
selector_chi2 = SelectKBest(score_func=chi2, k='all')
X_new = selector.fit_transform(X_select, y_meta_f)
X_new_mut = selector_mut.fit_transform(X_select, y_meta_f)
X_new_chi2 = selector_chi2.fit_transform(X_select, y_meta_f)
#meta_features_trainについて
#feature_names_boost = ["lgb", "cb1", "cb2", "svm"]
feature_names_boost = ["lgb", "cb1", "cb2", "svm", "rf"]
selector_meta = SelectKBest(score_func=f_classif, k='all')
selector_mut_meta = SelectKBest(score_func=mutual_info_classif, k='all')
selector_chi2_meta = SelectKBest(score_func=chi2, k='all')
X_meta = selector_meta.fit_transform(X_meta, y_meta_f)
X_meta_mut = selector_mut_meta.fit_transform(X_meta, y_meta_f)
X_meta_chi2 = selector_chi2_meta.fit_transform(X_meta, y_meta_f)
#train_diffについて
selector_diff = SelectKBest(score_func=f_classif, k='all')
selector_mut_diff = SelectKBest(score_func=mutual_info_classif, k='all')
selector_chi2_diff = SelectKBest(score_func=chi2, k='all')
X_diff = selector_diff.fit_transform(X_diff, y_meta_f)
X_diff_mut = selector_mut_diff.fit_transform(X_diff, y_meta_f)
X_diff_chi2 = selector_chi2_diff.fit_transform(X_diff, y_meta_f)
# スコアの確認
#X_select
scores = selector.scores_
scores_mut = selector_mut.scores_
scores_chi2 = selector_chi2.scores_
pvalues = selector_chi2.pvalues_
#meta_features_train
scores_meta = selector_meta.scores_
scores_mut_meta = selector_mut_meta.scores_
scores_chi2_meta = selector_chi2_meta.scores_
pvalues_meta = selector_chi2_meta.pvalues_
#train_diff
scores_diff = selector_diff.scores_
scores_mut_diff = selector_mut_diff.scores_
scores_chi2_diff = selector_chi2_diff.scores_
pvalues_diff = selector_chi2_diff.pvalues_

for name, score in zip(feature_names, scores):
    print(f"{name}: F-score = {score:.4f}")
    #print(f"{name}: MI-score = {score:.4f}")
mean_score = np.mean(scores)
print(f"\nF-scoreの平均: {mean_score:.4f}")
# スコアが高い特徴だけを残す（例：F-score > 1.0）
print("score:",score)
max_score = np.max(scores)
remove_feature = 'min1'
feature_names_rem = np.array(feature_names)
remove_index = np.where(feature_names_rem == remove_feature)[0]
min_score = scores[remove_index]
feature_names_rem = feature_names_rem

print("index",remove_index)
if max_score - min_score >= 10:
    # 対応する特徴量とスコアを除去
    scores_rem = np.delete(scores, remove_index)
    feature_names_rem = np.delete(feature_names_rem, remove_index)
    X_select_rem = np.delete(X_select, remove_index, axis=1)  # 特徴量行列からも列削除
    mask_rem = scores_rem > 5
    selected_features = [f for f, m in zip(feature_names_rem, mask_rem) if m]
    X_selected = X_select_rem[:, mask_rem]

else:
    mask = scores > 5
    selected_features = [f for f, m in zip(feature_names, mask) if m]
    X_selected = X_select[:, mask]
dataset_size = len(selected_features)
print("選ばれた特徴量の数：",dataset_size)
print("選ばれた評価指標:", selected_features)
print("X_selected",X_selected)

print("meta_features_train:",X_train_meta)

#テストデータで予測
#for model_idx, train_func in enumerate([train_lgb, train_xgb, train_cb, train_cb2, train_svm, train_rf]):
for model_idx, train_func in enumerate([train_lgb, train_cb, train_cb2, train_svm, train_rf]):
#for model_idx, train_func in enumerate([train_lgb, train_cb, train_cb2, train_svm]):
#for model_idx, train_func in enumerate([train_lgb, train_cb2, train_svm, train_rf]):
    if model_idx == 0:
        final_model = train_func(X_train, y_train,X_test= X_test)
        meta_features_test[:, model_idx] = final_model
        print("test_lbg:",meta_features_test[:,model_idx])
    else:  # 123
        final_model, model_proba = train_func(X_train, y_train, X_test=X_test)
        meta_features_test[:, model_idx] = final_model[:, 1]   # 確率値（クラス1）

sum_best_th_test = 0
sum1_test = 0

sum1_test = meta_features_test.sum(axis=1)#各行の合計値を調べる
row_max_test = np.max(meta_features_test, axis=1)
row_min_test = np.min(meta_features_test[:, :4], axis=1)
diff_test = row_max_test - row_min_test
mean_test = np.mean(meta_features_test, axis=1)
sum1_mean_test = sum1_test[sum1_test != 0].mean()

#print("row_min:", row_min)

#ここからmeta_features_test1
meta_features_test1[:,0] = sum1_test
meta_features_test1[:,1] = row_max_test
meta_features_test1[:,2] = row_min_test
meta_features_test1[:,3] = mean_test

X_select_test = meta_features_test1
if max_score - min_score >= 10:
    X_select_test = np.delete(X_select_test, remove_index, axis=1)  # 特徴量行列からも列削除
    X_selected_test = X_select_test[:, mask_rem]
else:
    X_selected_test = X_select_test[:, mask]
y = y_test
print(X_selected_test)

#箱ひげ図
df_plot = pd.DataFrame(meta_features_test1, columns=feature_names)
df_plot["y_test"] = y_test.reset_index(drop=True)
for col in feature_names:
    plt.figure()
    sns.boxplot(x="y_test", y=col, data=df_plot)
    plt.title(f"{col} by class")
    plt.savefig(os.path.join(output_boxplot,f"{col}_test.png"))
    plt.clf

#plot
for i in range(5):
    plt.figure()
    plt.title(f"{feature_names[i]} vs y_test")
    plt.xlabel("y_test")
    plt.ylabel(feature_names[i])
    plt.scatter(y_test_fil, meta_features_test1[:, i], alpha=0.5)
    plt.grid(True)
    plt.savefig(os.path.join(output_plot,f"{i}_test.png"))
    plt.clf


for model_idx in range(list_size-1):
    thresholds_conf_test = best_th2(y_test, meta_features_test[:,model_idx],model_idx)
    print("thresholds_test:",thresholds_conf_test)
    for i in range(len(X_test)):
        if thresholds_conf_test - 0.3 > meta_features_test[i, model_idx]:
            test_diff[i, model_idx] = 1
        elif thresholds_conf_test - 0.3 <= meta_features_test[i, model_idx] < thresholds_conf_test-0.2:
            test_diff[i, model_idx] = 5
        elif thresholds_conf_test - 0.2 <= meta_features_test[i, model_idx] < thresholds_conf_test-0.1:
            test_diff[i, model_idx] = 10
        elif thresholds_conf_test - 0.1 <= meta_features_test[i, model_idx] < thresholds_conf_test:
            test_diff[i, model_idx] = 15
        elif thresholds_conf_test <= meta_features_test[i, model_idx]:
            test_diff[i, model_idx] = 30
        else:
            test_diff[i, model_idx] = 0
    print("train_diff:",train_diff)
    print("test_diff:", test_diff)
# CSVファイルとして保存する
meta_features_train_df = pd.DataFrame(meta_features_train)
meta_features_test_df = pd.DataFrame(meta_features_test)
meta_features_train_df.to_csv(os.path.join(output_folder,'meta_features_train.csv'))
meta_features_test_df.to_csv(os.path.join(output_folder,'meta_features_test.csv'))

# train_diff と test_diff の値の分布を可視化
plt.figure(figsize=(12, 5))
plt.hist(meta_features_train1.flatten(), bins=30, alpha=0.5, label="train_diff", color="blue")
plt.hist(meta_features_test1.flatten(), bins=30, alpha=0.5, label="test_diff", color="red")
plt.legend()
plt.title("Train vs Test Feature Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.clf()

# メタモデルの学習と予測（LightGBMを使用）0になる確率を取得
params = {
    'iterations': 405,
    'learning_rate':0.01,
    'depth': 4,
    'od_type': 'IncToDec',
    'loss_function': 'Logloss',
    'od_wait':30,
    'random_state':42
    }


#回帰直線の作成
regressor_params = {
    "iterations": 100,
    "learning_rate": 0.1,
    "depth": 6,
    "verbose": False
}

#予測

#############################################################################################################
# 学習X_selected
#X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(X_selected, y_train, test_size=0.1,random_state=42)
train_pool = Pool(data=meta_features_train1, label=y_train_meta,feature_names=feature_names)
val_pool = Pool(data=meta_features_val1, label=y_val_meta)
print(y_train_meta)
print(y_val_meta)
#検証データに関する学習・および結果の確認
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=val_pool,early_stopping_rounds=params['od_wait'])    
probabilities_bal = model.predict_proba(meta_features_val1)[:,1]
binary_bal = model.predict(meta_features_val1)
# 特徴量の重要度を表示
importances = model.get_feature_importance(train_pool)
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("CatBoost Feature Importance")
plt.savefig(os.path.join(output_folder,"feature_imp.png"))
plt.clf()

#確認
print("X_train_sel:",meta_features_train1)
print("X_val_sel:",meta_features_val1)
print("final_probabilties",probabilities_bal)
fpr, tpr, thresholds = roc_curve(y_val_meta, probabilities_bal)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_sel_val = thresholds[youden_index]
print("最適な閾値:",best_threshold_sel_val)
# しきい値以上を 1、それ以外を 0 に分類
final_prediction_bal = (probabilities_bal >= best_threshold_sel_val).astype(int)
print("正解：", y_val_meta)
print("最終的な予測:",final_prediction_bal)
sum_val_sel_pro = probabilities_bal.sum()
mean_val_sel_pro = probabilities_bal.mean()
sum_val_sel = final_prediction_bal.sum()
mean_val_sel = final_prediction_bal.mean()
#model_idx = 10#meta_features_train2
model_idx = 50#X_select(valに対する予測結果)
#model_idx = 0#LightGBM
confusion(y_val_meta, final_prediction_bal ,model_idx)
id = 22
#学習曲線
history = model.get_evals_result()
print(history)
# グラフにプロットする
train_metric = history['learn']['Logloss']
eval_metric = history['validation']['Logloss']
plt.figure(figsize=(8, 6))
plt.plot(train_metric, label='train metric')
plt.plot(eval_metric, label='eval metric')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "learning_curve_val_sel.png"))
plt.close()

#テストデータに関する結果の確認
# 予測確率の取得
final_probabilities_select = model.predict_proba(meta_features_test1)[:,1]
final_binary_select = model.predict(meta_features_test1)

#確認
print("meta_features_train:",X_train_meta)
print("meta_features_test:",meta_features_test1)
print("final_probabilties",final_probabilities_select)
fpr, tpr, thresholds = roc_curve(y_test, final_probabilities_select)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_select = thresholds[youden_index]
print("最適な閾値:",best_threshold_select)
# しきい値以上を 1、それ以外を 0 に分類
final_prediction_select = (final_probabilities_select >= best_threshold_select).astype(int)
print("正解：", y_test)
print("最終的な予測:",final_prediction_select)
sum_test_sel_pro = final_probabilities_select.sum()
mean_test_sel_pro = final_probabilities_select.mean()
sum_test_sel = final_prediction_select.sum()
mean_test_sel = final_prediction_select.mean()
#model_idx = 10#meta_features_train2
model_idx = 40#X_select
#model_idx = 0#LightGBM
confusion(y_test, final_prediction_select ,model_idx)

##############################################################################################################
#学習train_diff
#検証データに関する学習・および結果の確認
train_pool = Pool(data=train_diff, label=y_train_meta,feature_names=feature_names)
val_pool = Pool(data=val_diff, label=y_val_meta)
diff_model = CatBoostClassifier(**params)
diff_model.fit(train_pool, eval_set=val_pool,early_stopping_rounds=params['od_wait'])    
probabilities_diff_val = diff_model.predict_proba(val_diff)[:,1]
binary_bal = diff_model.predict(val_diff)
#確認
print("X_train_diff:",train_diff)
print("X_val_diff:",val_diff)
print("final_probabilties",probabilities_diff_val)
fpr, tpr, thresholds = roc_curve(y_val_meta, probabilities_diff_val)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_bal = thresholds[youden_index]
print("最適な閾値:",best_threshold_bal)
# しきい値以上を 1、それ以外を 0 に分類
final_prediction_bal = (probabilities_bal >= best_threshold_bal).astype(int)
print("正解：", y_val_meta)
print("最終的な予測:",final_prediction_bal)
sum_val_diff_pro = probabilities_diff_val.sum()
mean_val_diff_pro = probabilities_diff_val.mean()
sum_val_diff = final_prediction_bal.sum()
mean_val_diff = final_prediction_bal.mean()
#model_idx = 10#meta_features_train2
model_idx = 65#train_diff(valに対する予測結果)
#model_idx = 0#LightGBM
confusion(y_val_meta, final_prediction_bal ,model_idx)
id = 22
#学習曲線
history = diff_model.get_evals_result()
print(history)
# グラフにプロットする
train_metric = history['learn']['Logloss']
eval_metric = history['validation']['Logloss']
plt.figure(figsize=(8, 6))
plt.plot(train_metric, label='train metric')
plt.plot(eval_metric, label='eval metric')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "learning_curve_val_diff.png"))
plt.close()

final_probabilities_diff = diff_model.predict_proba(test_diff)[:,1]
final_binary_diff = diff_model.predict(test_diff)
fpr, tpr, thresholds = roc_curve(y_test, final_probabilities_diff)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_all_diff = thresholds[youden_index]
print("最適な閾値:",best_threshold_all_diff)
# 予測確率の取得
print("train_diff:",train_diff)
print("test_diff:",test_diff)
print("final_probabilties_diff",final_probabilities_diff)
#final_prediction = meta_model.predict(final_test)
final_prediction_diff = (final_probabilities_diff >= best_threshold_all_diff).astype(int)
sum_test_diff_pro = final_probabilities_diff.sum()
mean_test_diff_pro = final_probabilities_diff.mean()
sum_test_diff = final_prediction_diff.sum()
mean_test_diff = final_prediction_diff.mean()
print("正解：", y_test)
print("最終的な予測:",final_prediction_diff)
model_idx = 15#train_diff
#model_idx = 0 #LightGBM
confusion(y_test, final_prediction_diff ,model_idx)
######################################################################################################################
# 学習meta_features_train
train_pool = Pool(data=X_train_meta, label=y_train_meta)
val_pool = Pool(data=X_val_meta, label=y_val_meta)

#検証データに関する学習・および結果の確認
model_meta = CatBoostClassifier(**params)
model_meta.fit(train_pool, eval_set=val_pool,early_stopping_rounds=params['od_wait'])    
probabilities_val_meta = model_meta.predict_proba(X_val_meta)[:,1]
binary_val_meta = model_meta.predict(X_val_meta)

#確認
print("X_train_sel:",X_train_meta)
print("X_val_sel:",X_val_meta)
print("final_probabilties",probabilities_val_meta)
fpr, tpr, thresholds = roc_curve(y_val_meta, probabilities_val_meta)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_val_meta = thresholds[youden_index]
print("最適な閾値:",best_threshold_val_meta)
# しきい値以上を 1、それ以外を 0 に分類
final_prediction_val_meta = (probabilities_val_meta >= best_threshold_val_meta).astype(int)
print("正解：", y_val_meta)
print("最終的な予測:",final_prediction_val_meta)
sum_val_meta_pro = probabilities_val_meta.sum()
mean_val_meta_pro = probabilities_val_meta.mean()
sum_val_meta = final_prediction_val_meta.sum()
mean_val_meta = final_prediction_val_meta.mean()
#model_idx = 10#meta_features_train2
model_idx = 53#meta_features_train(valに対する予測結果)
#model_idx = 0#LightGBM
confusion(y_val_meta, final_prediction_val_meta ,model_idx)
id = 22
#学習曲線
history = model_meta.get_evals_result()
print(history)
# グラフにプロットする
train_metric = history['learn']['Logloss']
eval_metric = history['validation']['Logloss']
plt.figure(figsize=(8, 6))
plt.plot(train_metric, label='train metric')
plt.plot(eval_metric, label='eval metric')
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "learning_curve_val_meta.png"))
plt.close()

#テストデータに関する結果の確認
# 予測確率の取得
final_probabilities_meta = model_meta.predict_proba(meta_features_test)[:,1]
final_binary_meta = model_meta.predict(meta_features_test)

#確認
print("meta_features_train:",meta_features_train)
print("meta_features_test:",meta_features_test)
print("final_probabilties",final_probabilities_meta)
fpr, tpr, thresholds = roc_curve(y_test, final_probabilities_meta)
# Youden's Indexを用いて最適なしきい値を決定
youden_index = np.argmax(tpr - fpr)
best_threshold_meta = thresholds[youden_index]
print("最適な閾値:",best_threshold_meta)
# しきい値以上を 1、それ以外を 0 に分類
final_prediction_meta = (final_probabilities_meta >= best_threshold_meta).astype(int)
print("正解：", y_test)
print("最終的な予測:",final_prediction_meta)
sum_test_meta_pro = final_probabilities_meta.sum()
mean_test_meta_pro = final_probabilities_meta.mean()
sum_test_meta = final_prediction_meta.sum()
mean_test_meta = final_prediction_meta.mean()
model_idx = 43#meta_features_train
confusion(y_test, final_prediction_meta ,model_idx)

# Fスコアと選ばれた特徴量を.txtに保存
output_path = os.path.join(output_folder,"f_score_output.txt")

#最終的に選ぶ手法を決定
if sum_test_sel - sum_val_sel >= 18:
    if mean_test_meta >= mean_test_diff:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_meta)
        model_idx = 100#final(diff)
        confusion(y_test, final_prediction_meta ,model_idx)
    elif mean_test_diff >= mean_test_meta:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_diff)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_diff ,model_idx)
elif mean_test_meta >= 0.9:#平均値が高すぎる．予測が1によりすぎてしまう可能性があるため除外
    if mean_test_diff >= mean_test_sel:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_diff)
        model_idx = 100#final(diff)
        confusion(y_test, final_prediction_diff ,model_idx)
    elif mean_test_sel >= mean_test_diff:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_select)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_select ,model_idx)
elif mean_test_diff >= 0.9:
    if mean_test_meta >= mean_test_sel:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_meta)
        model_idx = 100#final(diff)
        confusion(y_test, final_prediction_meta ,model_idx)
    elif mean_test_sel >= mean_test_meta:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_select)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_select ,model_idx)
elif mean_test_sel >= 0.9:
    if mean_test_meta >= mean_test_diff:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_meta)
        model_idx = 100#final(diff)
        confusion(y_test, final_prediction_meta ,model_idx)
    elif mean_test_diff >= mean_test_meta:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_diff)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_diff ,model_idx)
else:
    if mean_test_meta >= mean_test_diff and mean_test_meta >= mean_test_sel:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_meta)
        model_idx = 100#final(diff)
        confusion(y_test, final_prediction_meta ,model_idx)
    elif mean_test_diff >= mean_test_meta and mean_test_diff >= mean_test_sel:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_diff)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_diff ,model_idx)
    elif mean_test_sel >= mean_test_meta and mean_test_sel >= mean_test_diff:
        print("正解：", y_test)
        print("最終的な予測:",final_prediction_select)
        model_idx = 100#final(sel)
        confusion(y_test, final_prediction_select ,model_idx)

end = time.time()
print(end-start)
#各データセットで取得された合計値や平均値をtxtに保存
output_path = os.path.join(output_folder,"sum_mean_output.txt")

with open(output_path, "w", encoding="utf-8") as f:
    print("各結果(0,1)の合計",file=f)
    print(f"diff_val = {sum_val_diff}", file=f)
    print(f"diff_test = {sum_test_diff}", file=f)
    print(f"select_val = {sum_val_sel}", file=f)
    print(f"select_test = {sum_test_sel}", file=f)
    print(f"meta_val = {sum_val_meta}", file=f)
    print(f"meta_test = {sum_test_meta}", file=f)
    print("各結果(確信度)の合計",file=f)
    print(f"diff_val = {sum_val_diff_pro}", file=f)
    print(f"diff_test = {sum_test_diff_pro}", file=f)
    print(f"select_val = {sum_val_sel_pro}", file=f)
    print(f"select_test = {sum_test_sel_pro}", file=f)
    print(f"meta_val = {sum_val_meta_pro}", file=f)
    print(f"meta_test = {sum_test_meta_pro}", file=f)
    print("各結果の平均(0,1)",file=f)
    print(f"diff_val = {mean_val_diff}", file=f)
    print(f"diff_test = {mean_test_diff}", file=f)
    print(f"select_val = {mean_val_sel}", file=f)
    print(f"select_test = {mean_test_sel}", file=f)
    print(f"meta_val = {mean_val_meta}", file=f)
    print(f"meta_test = {mean_test_meta}", file=f)
    print("各結果の平均(確信度)",file=f)
    print(f"diff_val = {mean_val_diff_pro}", file=f)
    print(f"diff_test = {mean_test_diff_pro}", file=f)
    print(f"select_val = {mean_val_sel_pro}", file=f)
    print(f"select_test = {mean_test_sel_pro}", file=f)
    print(f"meta_val = {mean_val_meta_pro}", file=f)
    print(f"meta_test = {mean_test_meta_pro}", file=f)
    print("各結果の最適なしきい値",file=f)
    print(f"diff_val = {best_threshold_bal}", file=f)
    print(f"diff_test = {best_threshold_all_diff}", file=f)
    print(f"select_val = {best_threshold_sel_val}", file=f)
    print(f"select_test = {best_threshold_select}", file=f)
    print(f"meta_val:{best_threshold_val_meta}",file=f)
    print(f"meta_test:{best_threshold_meta}",file=f)
    print(f"実行時間:{end-start}",file=f)

# コピー先のパスを指定
destination_path = os.path.join(output_folder, os.path.basename(current_script_path))
# スクリプトをコピー
shutil.copy(current_script_path, destination_path)
print(f"スクリプトが {destination_path} にコピーされました。")

