from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from logistic_regression import Logistic_Regression

breast_cancer_data = load_breast_cancer()


X = breast_cancer_data['data']
y = breast_cancer_data['target']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=42
                                                    )
LR = Logistic_Regression()

LR.fit(X_train, y_train)

y_predict = LR.predict(X_test)

print(classification_report(y_test,
                            y_predict,
                            target_names=['positive', 'negative'],
                            digits=3
                            )
      )
