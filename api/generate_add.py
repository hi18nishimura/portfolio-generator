
from api.get_api_output import process as get_api_output
from api.get_program_file import process as get_program_file
from api.get_file_list import process as get_file_list

def process(prj_id,text):
    file_list = get_file_list(prj_id)
    print("Step 1: file_list =", file_list)
    file_list_with_program = get_program_file(file_list)
    print("Step 2: file_list_with_program =", file_list_with_program)
    output = get_api_output(file_list_with_program,add_prompt=text)
    print("Step 3: output =", output)
    return output