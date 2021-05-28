# Telco-Churn-MachineLearning
Telc Churn prediction using Machine Learning technics 

Classifiers used in the pipeline

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    MLPClassifier(alpha=0.001, solver='lbfgs', learning_rate='adaptive', max_iter=1000), # Neuro Net
    DecisionTreeClassifier(max_depth=5),
    GaussianNB(),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    BaggingClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
]
