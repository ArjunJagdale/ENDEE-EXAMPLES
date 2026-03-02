from endee import Endee

client = Endee()

try:
    remaining = client.list_indexes()
    if remaining:
        for idx in remaining:
            print(f"Found index: {idx['name']}")
    
    else:
        print("No indexes found.")

except Exception as e:
    print(f"Error Listing indexes: {e}")