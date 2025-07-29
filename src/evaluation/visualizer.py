# src/evaluation/visualizer.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

class Visualizer:
    def __init__(self, config_path=None, output_dir="output/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()

    def plot_model_comparison(self, metrics_dict):
        names = list(metrics_dict.keys())
        accuracies = [metrics_dict[name]['accuracy'] for name in names]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=names, y=accuracies)
        plt.ylabel("Accuracy")
        plt.title("Model Comparison")
        plt.savefig(os.path.join(self.output_dir, "model_comparison.png"))
        plt.close()

    def plot_roc_curves(self, model_names, metrics_dict):
        plt.figure(figsize=(8, 6))
        for model in model_names:
            if 'roc_curve' in metrics_dict[model]:
                fpr, tpr = metrics_dict[model]['roc_curve']
                plt.plot(fpr, tpr, label=f"{model} (AUC = {metrics_dict[model]['auc']:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()

    def plot_feature_importance(self, model, feature_names, model_name):
        if not hasattr(model.model, "feature_importances_"):
            return

        importances = model.model.feature_importances_
        indices = importances.argsort()[::-1]
        sorted_features = [feature_names[i] for i in indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=sorted_features)
        plt.title(f"{model_name} Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_feature_importance.png"))
        plt.close()
