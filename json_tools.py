import json

def read_json(json_file_absolute_dir = None):
    if json_file_absolute_dir is None:
        print("Please check the json_file_dir")
        return None
    with open(json_file_absolute_dir,"r",encoding="utf-8") as f:
        json_info = json.load(f)
    #print("Json info is :")

    return json_info

def write_json(json_dict = None, json_file_absolute_dir = None):
    if json_file_absolute_dir is None:
        print("Please check the json_file_dir")
        return None
    if json_dict is None:
        print("Please check the json_dict")
        return None
    with open(json_file_absolute_dir, "w") as f:
        json.dump(json_dict, f)
    print("%s json file write successfully!"%json_file_absolute_dir)
