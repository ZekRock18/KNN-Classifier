import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Classified Data", index_col=0)
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=33)

print(df.head())
#print(df.info())
scaler.fit(df.drop('TARGET CLASS', axis=1))


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
#print(scaled_features)

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
#print(df_feat.head())

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.40, random_state=101)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(pred)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.xlabel('K')
plt.ylabel('Error Rate')
#plt.show()    