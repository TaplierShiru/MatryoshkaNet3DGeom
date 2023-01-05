from utils import convert_files
from math import ceil, floor
import PIL.Image
import numpy as np
import argparse


def load_image(temp, alpha_map_to=255):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if temp.mode == 'RGBA':
        alpha = temp.split()[-1]
        bg = PIL.Image.new("RGBA", temp.size, (alpha_map_to, alpha_map_to, alpha_map_to) + (255,))
        bg.paste(temp, mask=alpha)
        im = bg.convert('RGB').copy()
        bg.close()
        temp.close()
    else:
        im = temp.copy()
        temp.close()
    return im


def crop_image(img: PIL.Image, size: int, pad = 100, background_value=255, resample=PIL.Image.LANCZOS):
    img_np = np.array(img)
    bg_mask = np.min(img_np, axis=2)==background_value
    cols = np.flatnonzero(np.logical_not(np.min(bg_mask, axis=0)))
    rows = np.flatnonzero(np.logical_not(np.min(bg_mask, axis=1)))
    p = ceil(np.max([cols[-1]-cols[0], rows[-1]-rows[0]]) / 2)
    imp_padded = np.pad(img_np, ((pad, pad), (pad, pad), (0,0)), 'constant', constant_values=background_value)
    row_offset = pad+floor((rows[0]+rows[-1])/2)
    col_offset = pad+floor((cols[0]+cols[-1])/2)
    img_cropped = imp_padded[
        row_offset-p: row_offset+p+1, 
        col_offset-p: col_offset+p+1,
        :
    ]

    img_cropped = PIL.Image.fromarray(img_cropped).resize((size, size), resample=resample)
    return img_cropped


def make_crop_func(size):
    def croptox(path):
        pad = 100
        img = load_image(PIL.Image.open(path))
        img_cropped = crop_image(img, size)
        img_cropped.save(path[:-4] + '.%d.png' % size, 'PNG')
        pass    
    return croptox


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Crop images to certain size.')
    parser.add_argument('directory', type=str, help='Directory with PNG files.', default='.')
    parser.add_argument('-s', '--size', type=int, default=128, help='Target size of images (width in pixels).')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively traverses the directory.')

    args = parser.parse_args()

    convert_files(args.directory, '.png', make_crop_func(args.size), args.recursive, '.%s.png' % args.size)
    pass
