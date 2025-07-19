from api.get_api_output import process

with open("file_list.txt", "r") as f:
    file_list = [line.strip() for line in f if line.strip()]

result = process(file_list)
print(result["markdown"])
