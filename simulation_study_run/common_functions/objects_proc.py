from pathlib import Path
from csv import writer
from datetime import datetime
import ast


DATE_STRING_FORMAT = "%Y:%m:%d %H:%M:%S"


def write_data_from_object_to_csv_file(obj: object, attributes_names_list: list, filepath: Path):
    with open(filepath, "w+", newline='') as file:
        w = writer(file, lineterminator='\r')
        headers = list()
        values = list()
        for attribute_name in attributes_names_list:
            headers.append(attribute_name)
            if (type(getattr(obj, attribute_name)) is tuple) and (len(getattr(obj, attribute_name)) == 1) \
                    and (type(getattr(obj, attribute_name)[0]) is datetime):
                values.append(getattr(obj, attribute_name)[0].strftime(DATE_STRING_FORMAT))
            else:
                values.append(getattr(obj, attribute_name))
        w.writerows([headers, values])


def load_data_from_dict_to_objects(obj: object, load_data_dict: dict):
    for key, item in load_data_dict.items():
        if '\\' in item:
            setattr(obj, key, Path(item))
        elif ":" in item:
            setattr(obj, key, datetime.strptime(item.replace("-", ":"), DATE_STRING_FORMAT))
        else:
            try:
                setattr(obj, key, ast.literal_eval(item))
            except (Exception,):
                if isinstance(item, str):
                    setattr(obj, key, item)
                else:
                    raise Exception("Error: cannot parse variable value")
    return obj
