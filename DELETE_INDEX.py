from endee import Endee

client = Endee()

# List of indexes to delete
indexes_to_delete = [
    "MOTORS_json_rag"
]

print(f"Deleting Indexes{indexes_to_delete}")

for index_name in indexes_to_delete:
    try:
        client.delete_index(index_name)
        print(f"Deleted index: {index_name}")
    except Exception as e:
        print(f"Failed to delete {index_name}: {e}")

