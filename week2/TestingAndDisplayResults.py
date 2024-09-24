from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)

p_result = pd.DataFrame(classifier.predict_proba(X_test))
p_result.columns = classifier.classes_
print(p_result)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))