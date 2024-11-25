import numpy as np

def create_labels_and_train(users_df, products_df, model, batch_size, num_epochs):
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

