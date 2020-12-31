import numpy as np
import Sentiment

review_path, label_path = 'data/reviews.txt', 'data/labels.txt'
seq_len = 200
split_fraction = 0.8

reviews, labels = Sentiment.load_test(review_path, label_path)
# Preprocess Data

preprocessed_text, split_reviews = Sentiment.preprocess_text(reviews)
words = preprocessed_text.split()
reviews_encoded, encoded_vocab = Sentiment.encode_text(preprocessed_text, split_reviews)

encoded_labels = Sentiment.encode_labels(labels)
filtered_review_encoded, filtered_label_encoded = Sentiment.outlier_removal(reviews_encoded, encoded_labels)
padded_features = Sentiment.pad_features(filtered_review_encoded, seq_len)

# Generate training and test data
split_index = int(len(padded_features) * 0.8)
train_x, remaining_x = padded_features[:split_index], padded_features[split_index:]
train_y, remaining_y = filtered_label_encoded[:split_index], filtered_label_encoded[split_index:]
test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
# print out the shapes of your resultant feature data

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print(padded_features[:30, :10])

train_data, valid_data, test_data = Sentiment.generate_data_loaders(train_x, train_y, val_x, val_y, test_x, test_y)
vocab_size = len()
