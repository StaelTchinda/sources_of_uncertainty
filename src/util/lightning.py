
import os

def is_main_ddp_process() -> bool:
    # From https://github.com/Lightning-AI/lightning/issues/8563#issuecomment-887291110
    return os.getenv("LOCAL_RANK", '0') == '0'