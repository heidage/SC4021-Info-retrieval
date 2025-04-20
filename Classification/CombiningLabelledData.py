import pandas as pd

# Load the datasets
df1 = pd.read_csv('/Users/jaredog/Downloads/git code/SC4021-Info-retrieval/ClassificationNew/modified_comments_without_neutral.csv')  # Replace with your dataset file paths
df2 = pd.read_csv('/Users/jaredog/Downloads/git code/SC4021-Info-retrieval/ClassificationNew/cleaned_stock_data - LABEL_HERE.csv')

df1['sentiment'] = df1['sentiment'].replace({'positive': 1.0, 'negative': -1.0})
# Merge the datasets based on 'post_id' column
merged_df = pd.merge(df1, df2[['post_id', ' Label']], on='post_id', how='inner')


# If you want to append the 'label' column from df2 to df1
df1['label'] = merged_df[' Label']
df1 = df1[df1['label'] != 0.0]
df1 = df1.dropna(subset=['label'])


# Save the resulting dataframe to a new CSV file
df1.to_csv('appended_dataset.csv', index=False)

print("Datasets merged and saved to 'appended_dataset.csv'")
