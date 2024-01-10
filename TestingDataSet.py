import pandas as pd

# Read the CSV file
df = pd.read_csv('./dataset.csv')

# Filter positive and negative reviews
positive_reviews = df[df['review_score'] == 1]
negative_reviews = df[df['review_score'] == -1]

# Sample 5000 rows from each
positive_sample = positive_reviews.head(5000)
negative_sample = negative_reviews.head(5000)

# Combine the samples to create a balanced dataset
balanced_df = pd.concat([positive_sample, negative_sample])

# Create a new DataFrame with renamed columns
new_df = balanced_df[['review_text', 'review_score']]
new_df.columns = ['review', 'voted_up']

# Change -1 to 0 in the 'voted_up' column
new_df['voted_up'] = new_df['voted_up'].replace(-1, 0)

# Save the new DataFrame to a CSV file
new_df.to_csv('balanced_test_score.csv', index=False)
