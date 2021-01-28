"""
make_spacer
~~~~~~~~~~~

Create a black JPG that can be used as a spacer in a video.
"""
import numpy as np

import imgwriter as iw


filepath = 'spacer.jpg'
a = np.zeros((1, 1080, 1920), dtype=int)
cspace = 'FPG'
iw.save_image(filepath, a, cspace)