import matplotlib.pyplot as plt
import numpy as np

# Data from the results provided
models = ['BERT', 'RoBERTa', 'Naive Bayes']
precision_negative = [0.894, 0.891, 0.727]  # For negative class
precision_positive = [0.828, 0.826, 0.789]  # For positive class
recall_negative = [0.812, 0.811, 0.812]  # For negative class
recall_positive = [0.904, 0.900, 0.700]  # For positive class
f1_negative = [0.851, 0.849, 0.767]  # For negative class
f1_positive = [0.864, 0.862, 0.742]  # For positive class
weighted_f1 = [0.857, 0.855, 0.754]  # Weighted F1 Score

n = len(models)
index = np.arange(n)
bar_width = 0.2

# Plotting
plt.figure(figsize=(15, 10))

# Precision plot
plt.subplot(2, 2, 1)
plt.bar(index, precision_negative, bar_width, label='Negative', alpha=0.7)
plt.bar(index + bar_width, precision_positive, bar_width, label='Positive', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Precision per Class Across Models')
plt.xticks(index + bar_width / 2, models)
plt.legend()

# Recall plot
plt.subplot(2, 2, 2)
plt.bar(index, recall_negative, bar_width, label='Negative', alpha=0.7)
plt.bar(index + bar_width, recall_positive, bar_width, label='Positive', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Recall per Class Across Models')
plt.xticks(index + bar_width / 2, models)
plt.legend()

# F1 Score plot
plt.subplot(2, 2, 3)
plt.bar(index, f1_negative, bar_width, label='Negative', alpha=0.7)
plt.bar(index + bar_width, f1_positive, bar_width, label='Positive', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score per Class Across Models')
plt.xticks(index + bar_width / 2, models)
plt.legend()

# Weighted F1 Score plot
plt.subplot(2, 2, 4)
plt.bar(index, weighted_f1, bar_width, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Weighted F1 Score')
plt.title('Weighted F1 Score Comparison')
plt.xticks(index, models)

plt.tight_layout()
plt.show()
