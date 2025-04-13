import glob
import os

new = "raw_data"
os.makedirs(new, exist_ok=True)


folders = glob.glob("vox/attack/*")
for folder in folders:
    if "video_cloaking_log" in folder:
        continue
    folder_name = folder.split("/")[-1]
    print(folder_name)
    os.makedirs(f"{new}/{folder_name}", exist_ok=True)
    # os.makedirs(os.path.join(new, folder_name), exist_ok=True)
    # os.makedirs(f"{new}/{folder_name}/source_frames", exist_ok=True)
    # copy the source frames to the new directory
    os.system(f"cp -r {folder}/source_frames {new}/{folder_name}/")
    # copy target.jpg to the new directory
    os.system(f"cp {folder}/target.jpg {new}/{folder_name}")