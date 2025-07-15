
from api.get_api_output import process as get_api_output
from api.get_file_list import process as get_file_list

def process(prj_id):
    file_list = get_file_list(prj_id)
    output = get_api_output(file_list)
    return output