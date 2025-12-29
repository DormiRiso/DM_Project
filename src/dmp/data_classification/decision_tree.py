import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from .classification_utils import _plot_summary_subplot
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.tree._tree import TREE_LEAF

# Import delle utilità esistenti
try:
    from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name
except ImportError:
    def make_safe_descriptor_name(descriptors):
        if isinstance(descriptors, list):
            return "_".join(descriptors[:3]) if len(descriptors) > 3 else "_".join(descriptors)
        return str(descriptors)

def prune_duplicate_leaves(dt):
    """
    Prune leaves if both children are leaves and make the same decision.
    Adapted from the notebook code.
    """
    def is_leaf(inner_tree, index):
        return (inner_tree.children_left[index] == TREE_LEAF and 
                inner_tree.children_right[index] == TREE_LEAF)

    def prune_index(inner_tree, decisions, index=0):
        if not is_leaf(inner_tree, inner_tree.children_left[index]):
            prune_index(inner_tree, decisions, inner_tree.children_left[index])
        if not is_leaf(inner_tree, inner_tree.children_right[index]):
            prune_index(inner_tree, decisions, inner_tree.children_right[index])

        if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF

    decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()
    prune_index(dt.tree_, decisions)

def decision_tree_classifier(
    df_train,
    df_test,
    num_feats=None,
    cat_feats=None,
    target_col="Rating",
    max_depth=None,
    min_samples_split=80,
    min_samples_leaf=100,
    criterion='gini',
    print_metrics=True,
    make_plot=True,
    descriptors=None,
    check_baseline=True,
    random_state=42
):
    """
    Decision Tree Classifier implementation.
    """
    
    print("\n" + "="*80)
    print("DECISION TREE CLASSIFIER")
    print("="*80)
    
    # Initialize variables
    y_test = None
    dt = None
    train_acc = test_acc = None
    
    try:
        from dmp.data_classification.classification_utils import (
            prepare_target_column, 
            clean_and_process_data,
            make_metrics,
            run_baseline_analysis
        )
        
        # Prepare target column
        df_train_processed, actual_target = prepare_target_column(
            df_train, descriptors, target_col
        )
        df_test_processed, _ = prepare_target_column(
            df_test, descriptors, target_col
        )
        
        # Clean and process data
        X_train, y_train, X_test, df_train_clean, df_test_clean, feature_names = clean_and_process_data(
            df_train_processed,
            df_test_processed,
            num_feats,
            cat_feats,
            actual_target
        )
        
        # Check if we have test data
        if len(X_test) > 0:
            y_test = df_test_clean[actual_target]
        
    except ImportError:
        print("Warning: classification_utils not found, using simplified processing")
        # Simplified processing as fallback
        all_feats = []
        if num_feats:
            all_feats.extend(num_feats)
        if cat_feats:
            all_feats.extend(cat_feats)
        
        # Basic cleaning
        df_train_clean = df_train.dropna(subset=all_feats + [target_col])
        df_test_clean = df_test.dropna(subset=all_feats + [target_col])
        
        X_train = df_train_clean[all_feats].values
        y_train = df_train_clean[target_col]
        X_test = df_test_clean[all_feats].values
        if len(df_test_clean) > 0:
            y_test = df_test_clean[target_col]
        feature_names = all_feats
        actual_target = target_col
    
    # Check if we have data
    if len(X_train) == 0:
        print("Error: No training data after cleaning!")
        return None
    
    print(f"\nTraining samples: {len(X_train)}")
    if y_test is not None:
        print(f"Test samples: {len(y_test)}")
    else:
        print("Test samples: 0")
    print(f"Number of features: {len(feature_names)}")
    print(f"Target column: {actual_target}")
    
    # Initialize and train the Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state
    )
    
    print("\nTraining Decision Tree...")
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = dt.predict(X_train)
    
    # Check if we have test data for predictions
    if y_test is not None and len(X_test) > 0:
        y_test_pred = dt.predict(X_test)
    else:
        y_test_pred = None
        print("Warning: No test data available for evaluation")
    
    if print_metrics:
        print("\n" + "-"*50)
        print("DECISION TREE PERFORMANCE")
        print("-"*50)
        
        # Training metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        
        print(f"\nTraining Set:")
        print(f"  Accuracy:  {train_acc:.4f}")
        print(f"  F1-score:  {train_f1:.4f}")
        
        # Test metrics (only if we have test data)
        if y_test is not None and y_test_pred is not None:
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')
            
            print(f"\nTest Set:")
            print(f"  Accuracy:  {test_acc:.4f}")
            print(f"  F1-score:  {test_f1:.4f}")
            
            print(f"\nClassification Report (Test Set):")
            print(classification_report(y_test, y_test_pred))
        else:
            test_acc = None
            print("\nTest Set: No test data available")
        
        print(f"\nTree Statistics:")
        print(f"  Depth: {dt.tree_.max_depth}")
        print(f"  Number of nodes: {dt.tree_.node_count}")
        print(f"  Number of leaves: {dt.tree_.n_leaves}")
        
        # Feature importance
        print("\nFeature Importance:")
        zipped = zip(feature_names, dt.feature_importances_)
        zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
        for col, imp in zipped:
            if imp > 0.01:  # Only show features with importance > 1%
                print(f"  {col}: {imp:.4f}")
    
    # Baseline analysis (only if we have test data)
    if check_baseline and y_test is not None and len(y_test) > 0:
        try:
            run_baseline_analysis(dt, X_train, y_train, "Decision Tree")
        except Exception as e:
            print(f"\nWarning: Baseline analysis failed: {e}")
    else:
        print("\nSkipping baseline analysis (no test data)")
    
    # Generate plots (only if we have test data)
    if make_plot and y_test is not None and len(y_test) > 0:
        generate_dt_plots(
            dt, X_train, y_train, X_test, y_test, y_test_pred,
            feature_names, actual_target, descriptors,
            max_depth, min_samples_split, min_samples_leaf, criterion
        )
    elif make_plot:
        print("\nSkipping plots (no test data available)")
    
    # Return the trained model and metrics
    return {
        'model': dt,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'feature_importances': dict(zip(feature_names, dt.feature_importances_)),
        'tree_depth': dt.tree_.max_depth,
        'n_nodes': dt.tree_.node_count
    }


def generate_dt_plots(
    dt, X_train, y_train, X_test, y_test, y_pred,
    feature_names, target_name, descriptors,
    max_depth, min_samples_split, min_samples_leaf, criterion
):
    """
    Generate comprehensive plots for Decision Tree analysis.
    """
    # Create directory for plots
    desc_name = make_safe_descriptor_name(descriptors) if descriptors else str(target_name)
    safe_feat_str = "_".join(feature_names[:3]) + ("_etc" if len(feature_names) > 3 else "")
    
    base_dir = Path("figures/classification/decision_tree")
    out_dir = base_dir / safe_feat_str
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generazione plot Decision Tree per: {desc_name}")
    
    # 1. Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=sorted(np.unique(y_train).astype(str)),
        filled=True,
        impurity=True,
        rounded=True,
        max_depth=min(3, dt.tree_.max_depth)  # Limit depth for readability
    )
    plt.title(f"Decision Tree (First 3 levels)\nMax Depth: {max_depth}, Criterion: {criterion}")
    tree_plot_path = out_dir / f"dt_tree_{desc_name}.png"
    plt.savefig(tree_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Tree plot saved: {tree_plot_path}")

    # 2. Confusion Matrix
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    unique_labels = sorted(np.unique(y_test))

    
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[str(l) for l in unique_labels]
    )    

    # Plot della matrice
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix: Decision Tree\n(Target: {desc_name})")

    acc = accuracy_score(y_test, y_pred)
    # 'weighted': calcola la media pesata in base al numero di istanze per classe (utile se sbilanciato)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    stats_text = (
        f"Accuracy: {acc:.2%}\n"
        f"Precision: {prec:.2%}\n"
        f"Recall: {rec:.2%}"
    )
    
    # figtext scrive coordinate relative alla figura intera (0,0 basso-sx, 1,1 alto-dx)
    # y=0.02 posiziona il testo in fondo
    plt.figtext(0.5, 0.02, stats_text, horizontalalignment='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
    
    # Aggiusta i margini per non tagliare il testo in basso
    plt.subplots_adjust(bottom=0.2)

    cm_plot_path = out_dir / f"dt_confusion_{desc_name}.png"
    plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_plot_path}")
    
    # 3. ROC Curves (una per ogni output possibile)
    try:
        # Importazione locale per evitare errori se il file non è nel path
        from dmp.data_classification.classification_utils import _plot_separated_roc
        
        # Chiamata alla funzione condivisa
        # Nota: model_tag="decision_tree" creerà la cartella figures/classification/decision_tree/...
        _plot_separated_roc(
            model=dt,
            X_test=X_test,
            y_test=y_test,
            model_tag="decision_tree", 
            feature_names=feature_names,
            descriptors=descriptors,
            target_name=target_name
        )
    except ImportError:
        print("⚠️ Impossibile importare _plot_separated_roc. Assicurati che classification_utils sia accessibile.")
    except Exception as e:
        print(f"⚠️ Errore durante la generazione delle ROC curves: {e}")
    
    # 4. Precision-Recall Curve
    try:
        # Importazione locale se necessario, o assicurati sia importata in alto
        from dmp.data_classification.classification_utils import _plot_separated_precision_recall
        
        _plot_separated_precision_recall(
            model=dt,
            X_test=X_test,
            y_test=y_test,
            model_tag="decision_tree", 
            feature_names=feature_names,
            descriptors=descriptors,
            target_name=target_name
        )
    except ImportError:
        print("⚠️ Impossibile importare _plot_separated_precision_recall.")
    except Exception as e:
        print(f"⚠️ Errore generazione Precision-Recall curves: {e}")
    
    # 5. Feature Importance Bar Plot
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance - Decision Tree\n{desc_name}")
    plt.tight_layout()
    importance_plot_path = out_dir / f"dt_feature_importance_{desc_name}.png"
    plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved: {importance_plot_path}")
    
    # 6. Tree Complexity Analysis
    analyze_tree_complexity(
        X_train, y_train, X_test, y_test,
        feature_names, target_name, desc_name, out_dir
    )
    
    _plot_summary_subplot(model=dt,
            predictions=y_pred,
            X_test=X_test,
            y_test=y_test,
            model_tag="decision_tree", 
            feature_names=feature_names,
            descriptors=descriptors,
            target_name=target_name)
        
    # 7. Hyperparameter Tuning (opzionale - commenta se troppo lento)
    print("🔄 Performing hyperparameter tuning...")
    try:
        best_dt = perform_hyperparameter_tuning(
            X_train, y_train, X_test, y_test,
            feature_names, target_name, desc_name, out_dir
        )
        print(f"✅ Hyperparameter tuning completed")
    except Exception as e:
        print(f"⚠️  Hyperparameter tuning failed: {e}")
    
    print(f"🎉 All plots saved in: {out_dir}")


def analyze_tree_complexity(X_train, y_train, X_test, y_test, feature_names, target_name, desc_name, out_dir):
    """
    Analyze how tree complexity affects performance.
    """
    max_depths = list(range(1, 21))
    min_samples_leafs = [1, 5, 10, 20, 50, 100]
    
    train_accs_depth = []
    test_accs_depth = []
    nodes_depth = []
    
    train_accs_leaf = []
    test_accs_leaf = []
    nodes_leaf = []
    
    # Analyze by max_depth
    print("\nAnalyzing tree complexity by max_depth...")
    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        
        train_accs_depth.append(train_acc)
        test_accs_depth.append(test_acc)
        nodes_depth.append(dt.tree_.node_count)
    
    # Analyze by min_samples_leaf
    print("Analyzing tree complexity by min_samples_leaf...")
    for min_leaf in min_samples_leafs:
        dt = DecisionTreeClassifier(min_samples_leaf=min_leaf, random_state=42)
        dt.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        
        train_accs_leaf.append(train_acc)
        test_accs_leaf.append(test_acc)
        nodes_leaf.append(dt.tree_.node_count)
    
    # Plot complexity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Depth vs Accuracy
    axes[0, 0].plot(max_depths, train_accs_depth, 'b-', label='Train', marker='o')
    axes[0, 0].plot(max_depths, test_accs_depth, 'r-', label='Test', marker='o')
    axes[0, 0].set_xlabel("Max Depth")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs Max Depth")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Depth vs Nodes
    axes[0, 1].plot(max_depths, nodes_depth, 'g-', marker='o')
    axes[0, 1].set_xlabel("Max Depth")
    axes[0, 1].set_ylabel("Number of Nodes")
    axes[0, 1].set_title("Tree Size vs Max Depth")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min samples leaf vs Accuracy
    axes[1, 0].plot(min_samples_leafs, train_accs_leaf, 'b-', label='Train', marker='o')
    axes[1, 0].plot(min_samples_leafs, test_accs_leaf, 'r-', label='Test', marker='o')
    axes[1, 0].set_xlabel("Min Samples Leaf")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy vs Min Samples Leaf")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Min samples leaf vs Nodes
    axes[1, 1].plot(min_samples_leafs, nodes_leaf, 'g-', marker='o')
    axes[1, 1].set_xlabel("Min Samples Leaf")
    axes[1, 1].set_ylabel("Number of Nodes")
    axes[1, 1].set_title("Tree Size vs Min Samples Leaf")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f"Decision Tree Complexity Analysis\nTarget: {target_name}", fontsize=16)
    plt.tight_layout()
    complexity_path = out_dir / f"dt_complexity_analysis_{desc_name}.png"
    plt.savefig(complexity_path, dpi=150, bbox_inches='tight')
    plt.close()


def perform_hyperparameter_tuning(X_train, y_train, X_test, y_test, feature_names, target_name, desc_name, out_dir):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    print("\nPerforming hyperparameter tuning...")
    
    param_dist = {
        'max_depth': [None] + list(range(2, 20)),
        'min_samples_split': [2,5,10,25,30,50,75,80,100],
        'min_samples_leaf': [2,5,10,25,30,50,75,80,100],
        'criterion': ['gini', 'entropy']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        dt,
        param_distributions=param_dist,
        n_iter=500,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Evaluate best model on test set
    best_dt = random_search.best_estimator_
    test_acc = accuracy_score(y_test, best_dt.predict(X_test))
    print(f"Test accuracy with best model: {test_acc:.4f}")
    
    # Plot hyperparameter importance
    results_df = pd.DataFrame(random_search.cv_results_)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Max Depth vs Score
    depth_results = results_df.groupby('param_max_depth')['mean_test_score'].mean()
    axes[0, 0].plot(depth_results.index.astype(str), depth_results.values, 'b-', marker='o')
    axes[0, 0].set_xlabel("Max Depth")
    axes[0, 0].set_ylabel("Mean CV Score")
    axes[0, 0].set_title("Max Depth vs Performance")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Min Samples Split vs Score
    split_results = results_df.groupby('param_min_samples_split')['mean_test_score'].mean()
    axes[0, 1].plot(split_results.index.astype(str), split_results.values, 'g-', marker='o')
    axes[0, 1].set_xlabel("Min Samples Split")
    axes[0, 1].set_ylabel("Mean CV Score")
    axes[0, 1].set_title("Min Samples Split vs Performance")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min Samples Leaf vs Score
    leaf_results = results_df.groupby('param_min_samples_leaf')['mean_test_score'].mean()
    axes[1, 0].plot(leaf_results.index.astype(str), leaf_results.values, 'r-', marker='o')
    axes[1, 0].set_xlabel("Min Samples Leaf")
    axes[1, 0].set_ylabel("Mean CV Score")
    axes[1, 0].set_title("Min Samples Leaf vs Performance")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Criterion vs Score
    criterion_results = results_df.groupby('param_criterion')['mean_test_score'].mean()
    axes[1, 1].bar(criterion_results.index.astype(str), criterion_results.values, color=['blue', 'green'])
    axes[1, 1].set_xlabel("Criterion")
    axes[1, 1].set_ylabel("Mean CV Score")
    axes[1, 1].set_title("Criterion vs Performance")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Hyperparameter Tuning Results\nTarget: {target_name}", fontsize=16)
    plt.tight_layout()
    tuning_path = out_dir / f"dt_hyperparameter_tuning_{desc_name}.png"
    plt.savefig(tuning_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_dt
