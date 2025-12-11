import requests
from tqdm import tqdm
import os

def download(url, out_path):
    # If file exists → resume download
    resume_header = {}
    pos = 0
    if os.path.exists(out_path):
        pos = os.path.getsize(out_path)
        resume_header = {'Range': f'bytes={pos}-'}

    response = requests.get(url, stream=True, headers=resume_header)
    
    # If server supports range requests, calculate total remaining size
    total = int(response.headers.get('content-length', 0))
    total += pos  # for tqdm total size display

    mode = 'ab' if pos > 0 else 'wb'

    with open(out_path, mode) as file, tqdm(
        initial=pos,
        total=total,
        unit='B',
        unit_scale=True,
        desc="Downloading"
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = file.write(data)
            bar.update(size)


url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
out_file = "dev-clean.tar.gz"

download(url, out_file)
