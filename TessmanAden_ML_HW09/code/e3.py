
from email import header
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,ConfusionMatrixDisplay, classification_report


columns = ["age of the patient","spectacle prescription","astigmatic", "tear production rate"]
classes = ['hard contact lenses', 'soft contact lenses', 'no contact lenses']
data = pd.read_csv('lenses.csv', header = None)


X = data.iloc[: , 1:-1]# exlcude frist col as it is an extra index
y = data.iloc[: , -1]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state=0)

# Train Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)

# text representations
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
print('\nAccuracy score: ', accuracy)
print(f'Important Features: {dt.feature_importances_}')


# Vistual Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=dt.classes_)
disp.plot()
plt.show()

# dt text representations
text_representation = export_text(dt)
print(text_representation)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=columns, class_names=classes, filled=True, rounded=True)
plt.show()

