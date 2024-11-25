#file to collect functions related to model training and evaluation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def create_labels_and_train(users_df, products_df, model, batch_size, num_epochs):
    for epoch in range(num_epochs):
        # initilize the target similarity for the batch
        target_similarity = []

        # we're making the target similarity balanced, so there's an equal number of posivie and negetive indices in each batch
        num_indices = batch_size // 2

        # generating 1/2 batch size of random pairs, where there are positive indices (user and product have the same ID)
        positive_user_indices = np.random.randint(0, len(users_df), size=num_indices)
        # initialize storage of positive indicies
        positive_product_indices = []
        # loop over every user
        for user_idx in positive_user_indices:
            # locating product IDs in the user dataframe for the user we sampled
            user_product_id = users_df.iloc[user_idx]['product_id']
            # finding matching products in the products dataframe
            matching_products = products_df[products_df['product_id'] == user_product_id]
            # append the matching product to the positive product indices
            positive_product_indices.append(matching_products.index[0])

        # Generate random negative pairs (user and product have different product_ids)
        negative_user_indices = np.random.randint(0, len(users_df), size=num_indices)
        #print("NEGATIVE USER INDICES: ", negative_user_indices)
        negative_product_indices = []
        for user_idx in negative_user_indices:
            user_product_id = users_df.iloc[user_idx]['product_id']
            # find a product that doesn't have a matching product id
            non_matching_products = products_df[products_df['product_id'] != user_product_id]
            # append that to the negetive indicies
            negative_product_indices.append(non_matching_products.sample(1).index[0])

        # combining both positive and negetive indicies
        user_indices = np.concatenate([positive_user_indices, negative_user_indices])
        product_indices = np.concatenate([positive_product_indices, negative_product_indices])

        # create target similarity labels for the positive and negetive pairs
        target_similarity.extend([1] * num_indices)  # Positive pairs
        target_similarity.extend([0] * num_indices)  # Negative pairs
        target_similarity = np.array(target_similarity)

        # get the positive & negetive user data
        user_data = users_df.iloc[user_indices]
        user_ids = user_data['user_id'].tolist()
        product_data = products_df.iloc[product_indices]
        item_ids = product_data['product_id'].tolist()

        user_data = user_data.drop(columns=['product_id', 'user_id'])
        product_data = product_data.drop(columns=['product_id', 'flattened_name_embedding', 'flattened_brand_embedding'])

        # Split data into training and testing sets
        X_train_users, X_test_users, X_train_products, X_test_products, y_train, y_test = train_test_split(
            user_data, product_data, target_similarity, test_size=0.2, random_state=42
        )

        # Train the model on the training data
        model.fit([X_train_users, X_train_products], y_train, epochs=1, batch_size=batch_size, verbose=False)

        # Predict on the test data
        predicted_probabilities = model.predict([X_test_users, X_test_products]).flatten()

        # Convert probabilities to binary predictions
        y_pred = (predicted_probabilities > 0.5).astype(int)

        # Evaluate the model on the test data
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, predicted_probabilities)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}, ROC AUC: {auc:.4f}")
