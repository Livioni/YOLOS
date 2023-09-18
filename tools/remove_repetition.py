import os

def remove_duplicate_lines_ordered(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(unique_lines)

# 使用函数
input_file = 'path_to_your_txt_file.txt'
output_file = 'path_to_output_file.txt'



folder_path = '/Volumes/Livion/Pandas_coco/labels'
file_list = os.listdir(folder_path)
for file in file_list:
    if file == '.DS_store':
        continue
    input_file = os.path.join(folder_path, file)
    remove_duplicate_lines_ordered(input_file, input_file)
    print('complete: ', file)