# Made by Jim.Wang V1 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable




from .HailuoTarot import TarotDealCard,Txt2Img

NODE_CLASS_MAPPINGS = {
    "TarotDealCard":TarotDealCard,
    "txt2img":Txt2Img
}


print('\033[34mHailuoTarot TarotDealCard Nodes: \033[92mLoaded\033[0m')