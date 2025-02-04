import torch
import os
from pathlib import Path
import urllib.request


def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Device count:", torch.cuda.device_count())
        print("Current device index:", torch.cuda.current_device())
        print(
            "Current device name:",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
    else:
        print("CUDA is not available!")


def get_sample_docs(DATA_PATH: str):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    try:
        urllib.request.urlretrieve(
            "https://aclanthology.org/2020.findings-emnlp.92.pdf",
            f"{DATA_PATH}/PhoBERT - Pretrained Language Models for Vietnamese.pdf",
        )
        urllib.request.urlretrieve(
            "https://arxiv.org/pdf/2408.00118",
            f"{DATA_PATH}/Gemma 2 - Improving Open Language Models at a Practical Size.pdf",
        )
        urllib.request.urlretrieve(
            "https://arxiv.org/pdf/2403.08295",
            f"{DATA_PATH}/Gemma - Open Models Based on Gemini Research and Technology.pdf",
        )
        urllib.request.urlretrieve(
            "https://arxiv.org/pdf/2412.15115",
            f"{DATA_PATH}/Qwen2.5 - Technical Report.pdf",
        )
        # urllib.request.urlretrieve(
        #     "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15.pdf",
        #     f"{DATA_PATH}/LUẬT TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ.pdf",
        # )
        # urllib.request.urlretrieve(
        #     "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf",
        #     f"{DATA_PATH}/VĂN BẢN QUY PHẠM PHÁP LUẬT.pdf",
        # )
    except Exception as e:
        print(f"Failed to download the file: {e}")
