import cv2
import numpy as np
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt

from bing_image_downloader import downloader

downloader.download(
    "cloony",
    limit=10000,
    output_dir='dataset',
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)