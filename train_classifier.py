import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data directly as a NumPy array
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Preprocess the data to ensure consistent shapes (padding sequences if necessary)
max_sequence_length = max(len(seq) for seq in data_dict['data'])
data_padded = np.array([np.pad(seq, (0, max_sequence_length - len(seq)), 'constant') for seq in data_dict['data']])

data = np.asarray(data_padded)
labels = np.asarray(data_dict['labels'])

# Check class distribution
unique_classes, class_counts = np.unique(labels, return_counts=True)
print("Class Distribution:")
for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls}: {count} samples")

# Filter out classes with only one sample
valid_classes = [cls for cls, count in zip(unique_classes, class_counts) if count > 1]
mask = np.isin(labels, valid_classes)
filtered_data = data[mask]
filtered_labels = labels[mask]

# Use train_test_split for stratified splitting
x_train, x_test, y_train, y_test = train_test_split(filtered_data, filtered_labels, test_size=0.2, random_state=42, stratify=filtered_labels)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate and print the accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly'.format(score * 100))

# Save the trained model and labels
with open('model.p', 'wb') as file:
    pickle.dump({'model': model, 'labels': filtered_labels}, file)
