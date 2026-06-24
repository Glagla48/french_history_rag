import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from src.config import RAW_DATA_DIR, CLEAN_DATA_DIR

def clean_arrow(lines:list[str]):
    new_content = []
    for line in lines:
        if not "↑ " in line:
            new_content.append(line)
    return new_content

def clean_strange_thing(lines:list[str]):
    for i, line in enumerate(lines):
        lines[i] = line.replace("[modifier | modifier le code]", "")
    return lines

def cleanning_pipeline(original_folder:str|Path, 
                       new_folder:str|Path, 
                       filename:str):
    filepath = Path(original_folder) / filename
    new_file_path = Path(new_folder) / filename

    with open(filepath, "r") as f:
        text = f.readlines()

    res = clean_arrow(text)
    res = clean_strange_thing(res)

    with open(new_file_path, "w") as f:
        f.writelines(res)



    

def clean(original_folder:str|Path, 
        new_folder:str|Path):
    files = [f for f in os.listdir(original_folder)]

    args = [(original_folder, new_folder, f) for f in files]
    with ProcessPoolExecutor(max_workers=int(os.cpu_count()/2) +1) as executor:
        list(executor.map(cleanning_pipeline, args, chunksize=10))





    pass

if __name__ == "__main__":
    clean("./data/raw/french")