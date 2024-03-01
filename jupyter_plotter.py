"""
This is a (should-have-been-working-but-not-really) hack for Jupyter notebooks to display matplotlib figures with arbitrary backends inline.

(sigh...)

Fortunately, the agg backend still seems to work with LaTeX math mode,
so unless we have to do fancy stuff, we can keep just showing the figure.
"""


#from PIL import Image
#import numpy as np
#import os
#from matplotlib import pyplot as plt
#from pdf2image import convert_from_path


def plot_jupyter_figure(fig):
  pass
  #fig.show()
  ##fig.savefig('temp.pdf')
  ##convert_from_path('temp.pdf', 1, size=(1600, None))[0].save('temp.png','PNG')
  #fig.savefig('temp.png')
  #with Image.open('temp.png') as pil_im:
  #  im_array = np.asarray(pil_im)
  #  plt.imshow(im_array)
  #  plt.show()
  #os.remove('temp.png')
  #os.remove('temp.pdf')
