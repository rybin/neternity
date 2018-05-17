#!/usr/bin/env python3.6
from PIL import Image, ImageDraw, ImageFont, ImageChops
import string
from pathlib import Path
from random import random


ImgSize = 200
II = 16


def scale(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio
    and place it in center of white 'max_size' image
    """
    image.thumbnail(max_size, method)
    offset = (int((max_size[0] - image.size[0]) / 2),
              int((max_size[1] - image.size[1]) / 2))
    back = Image.new('L', max_size, color=(255))
    back.paste(image, offset)

    return back


def cropLetterImage(img):
    bg = Image.new('L', (ImgSize, ImgSize), color=(255))
    diff = ImageChops.difference(img, bg)
    return img.crop(diff.getbbox())


def letter(img):
    img = cropLetterImage(img)
    IMax = max(img.size[0], img.size[1])
    img = scale(img, (IMax, IMax))
    return img.resize((II, II))


def drawLetter(char, font):
    img = Image.new('L', (ImgSize, ImgSize), color=(255))
    d = ImageDraw.Draw(img)
    d.text((0, 0), char, font=font, fill=(0))
    return letter(img)


# fontsize = 100
# font = ImageFont.truetype('./font/DejaVuSansMono.ttf', fontsize)
# img = drawLetter('g', font)
# img.show()
# exit()

if __name__ == '__main__':
    # fonts = ['DejaVuSansMono.ttf', 'DejaVuSans.ttf', 'DejaVuSerif.ttf']
    # fontpath = './font/DejaVuSansMono.ttf'
    fontsize = 100

    fonts = Path('./font/')

    for fontfile in fonts.glob('*.ttf'):
        print(fontfile)
        font = ImageFont.truetype(str(fontfile), fontsize)
        for i in string.ascii_letters:
            img = drawLetter(i, font)
            img.save(f'pic/{i}_{fontfile.stem}_{str(random())[2:]}.png', 'png')

    # for j in fonts():
    #     # font = ImageFont.truetype(fontpath, fontsize)
    #     font = ImageFont.truetype('./font/' + j, fontsize)

    #     for i in string.ascii_letters:
    #         img = drawLetter(i, font)
    #         img.save('pic/' + i + '.png', 'png')
    #         # img.show()
