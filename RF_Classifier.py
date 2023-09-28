from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.best_params_ = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

    def calculate_AUPRC(self, y_true, y_scores):
        average_precision = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auc_score = auc(recall, precision)
        return average_precision, auc_score

    def evaluate(self, X_test, y_test):
        # Predict probabilities instead of classes
        y_scores = self.model.predict_proba(X_test)[:, 1]
        average_precision, auc_score = self.calculate_AUPRC(y_test, y_scores)
        print(f"Average Precision: {average_precision}")
        print(f"AUPRC: {auc_score}")
        self.plot_AUPRC(y_test, y_scores)

    def tune_hyperparameters(self, X_train, y_train, params_grid):
        grid_search = GridSearchCV(self.model, params_grid, cv=5, scoring='average_precision', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {self.best_params_}")
        print(f"Best score: {grid_search.best_score_}")

    def train(self, X, y):
        if self.best_params_:
            self.model.set_params(**self.best_params_)
        self.model.fit(X, y)

    def plot_AUPRC(self, y_true, y_scores):
        average_precision, _ = self.calculate_AUPRC(y_true, y_scores)

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        plt.show()



