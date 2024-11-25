#Utility file to collect all the functions defined for the generating recommendations functionality.

#Checks if a user exits in a dataframe
def check_user_exists(users_df, user_id):
    if user_id in users_df['user_id'].values:
        print(f"User ID {user_id} exists in the DataFrame.")
        return True
    else:
        print(f"User ID {user_id} does not exist in the DataFrame.")
        return False


#Given a product id, looks up the product name in the original dataframe and returns it
def lookup_product_name(product_ids, original_products):
    for id in product_ids:
        product_row = original_products[original_products['id'] == id]
        name = product_row['name']
        print(f"ID = {id}, Name = {name}")