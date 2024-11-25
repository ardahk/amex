import numpy as np

def create_labels_and_train1(users_df, products_df, model, batch_size, num_epochs):
    for epoch in range(num_epochs):
        # generate random user-item pairs through random indices for each batch
        user_indices = np.random.randint(0, len(users_df), size=batch_size)
        product_indices = np.random.randint(0, len(products_df), size=batch_size)

        # extract the data
        user_data = users_df.iloc[user_indices]#.copy()  # copy to avoid SettingWithCopyWarning
        product_data = products_df.iloc[product_indices]#.copy()

        # we will be creating target similarity labels
        target_similarity = []

        # loop through user and product indices to create labels
        for user_idx, product_idx in zip(user_indices, product_indices):
            user_product_id = users_df.iloc[user_idx]['product_id']
            item_product_id = products_df.iloc[product_idx]['product_id']

            # if the user and item product id match, it means the user purchased the product
            # otherwise, there is no interaction and the target similarity would be 0
            target_similarity.append(1 if user_product_id == item_product_id else 0)

        # convert to a numpy array
        target_similarity = np.array(target_similarity)

        # train the model with the pairs
        model.fit([user_data.values, product_data.values], target_similarity, epochs=1, batch_size=batch_size)


import numpy as np

def create_labels_and_train(users_df, products_df, model, batch_size, num_epochs, val_users_df, val_products_df):
    for epoch in range(num_epochs):
        # Generate random user-item pairs through random indices for each batch
        user_indices = np.random.randint(0, len(users_df), size=batch_size)
        product_indices = np.random.randint(0, len(products_df), size=batch_size)

        # Extract the data
        user_data = users_df.iloc[user_indices]
        product_data = products_df.iloc[product_indices]

        # Create target similarity labels
        target_similarity = []

        # Loop through user and product indices to create labels
        for user_idx, product_idx in zip(user_indices, product_indices):
            user_product_id = users_df.iloc[user_idx]['product_id']
            item_product_id = products_df.iloc[product_idx]['product_id']

            # If the user and item product id match, it means the user purchased the product
            # Otherwise, there is no interaction and the target similarity would be 0
            target_similarity.append(1 if user_product_id == item_product_id else 0)

        # Convert to a numpy array
        target_similarity = np.array(target_similarity)

        # Train the model with the pairs
        model.fit([user_data.values, product_data.values], target_similarity, epochs=1, batch_size=batch_size)

        # Evaluate the model's accuracy on the validation set
        val_user_indices = np.random.randint(0, len(val_users_df), size=batch_size)
        val_product_indices = np.random.randint(0, len(val_products_df), size=batch_size)

        # Extract validation data
        val_user_data = val_users_df.iloc[val_user_indices]
        val_product_data = val_products_df.iloc[val_product_indices]

        # Create target similarity labels for validation
        val_target_similarity = []

        for val_user_idx, val_product_idx in zip(val_user_indices, val_product_indices):
            val_user_product_id = val_users_df.iloc[val_user_idx]['product_id']
            val_item_product_id = val_products_df.iloc[val_product_idx]['product_id']
            
            # Similarity for validation set
            val_target_similarity.append(1 if val_user_product_id == val_item_product_id else 0)

        val_target_similarity = np.array(val_target_similarity)

        # Evaluate the model
        predictions = model.predict([val_user_data.values, val_product_data.values])
        predictions = (predictions > 0.5).astype(int)  # Convert to 0 or 1 (binary classification)

        # Calculate accuracy
        accuracy = np.mean(predictions == val_target_similarity)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')
