"""
    Gradient Gen
    jb, 2025-05-17

    This is supposed to generate multiple types of gradients to showcase the difference to how to interpolate between two or more colors
"""
#!/bin/python3
from PIL import Image, ImageColor, ImageDraw, ImageFont
import colorsys
import numpy as np
from scipy.special import comb
import argparse

stripWidthMultiplier = 2
stripHeight = 100
stripMargin = 20

#colorToInterp = ['#102eed', '#0aad15']
#colorToInterp = ['#FF0000', '#0000FF']
#colorToInterp = ['#000000', '#FFFFFF']
#colorToInterp = ['#FF0000', '#AAFFFF']

# \/\/\/ Internal Code, do not change

# https://stackoverflow.com/questions/45165452/how-to-implement-a-smooth-clamp-function-in-python
def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    result = np.clip(result, 0, 1)

    return result

def interpolate(val, _len, _type, *args, **kwargs):
    if _type == 'lin':
        interp = np.linspace(0, 1, _len)
    elif _type == 'smoothstep':
        interp = smoothstep(np.arange(_len), 0, _len, args[0])
    elif _type == 'quad':
        interp =  np.arange(_len)**2/(_len**2)
    elif _type == 'sin':
        x = np.arange(_len)
        interp = np.sin(x*np.pi/2*(1/_len))
    # loop stuff
    elif _type == 'loop_lin':
        interp = np.concatenate((np.linspace(0, 1, _len//2), np.linspace(1, 0, _len//2)))
    elif _type == 'loop_smoothstep':
        x = smoothstep(np.arange(_len//2), 0, _len//2, args[0])
        interp = np.concatenate(( x, 1-x))
    elif _type == 'loop_sin':
        x = np.arange(_len)
        interp = np.sin(x*np.pi*(1/_len))
    else:
        raise UserWarning(f"Invalid type {_type:s}")

    if 'wrapPath' in kwargs:
        wrapPath = kwargs['wrapPath']
    else:
        wrapPath = False

    delta = val[1] - val[0]
    if wrapPath and (1 > abs(delta) > 0.5):
        if delta < 0:
            interp *= 1-abs(delta)
        else:
            interp *= delta-1
    else:
        interp *= delta
    interp += val[0]
    print(val)

    # wrap values to between 0 and -1
    interp = np.where((interp > 1), interp - 1, interp)
    interp = np.where((interp < 0), interp + 1, interp)

    return interp

def getConversionSpaces(sp):
    if sp == 'rgb' or sp == None:
        return lambda *x: x, lambda *x: x
    elif sp == 'hsv':
        return colorsys.rgb_to_hsv, colorsys.hsv_to_rgb
    elif sp == 'hls':
        return colorsys.rgb_to_hls, colorsys.hls_to_rgb
    else:
        raise UserWarning(f"Invalid space {sp:s}")

def getYStart(yIdx):
    return stripMargin + yIdx*(2*stripMargin + stripHeight)

def getInterpolatedColors(colorA, colorB, colorSpConv, interp, _len):
    if type(interp) == str:
        interp = (interp, )
    convDefs = getConversionSpaces(colorSpConv)

    # convert 0-255 to 0-1
    colors = np.array([colorA, colorB], dtype=float)
    colors /= 255
    #print(colors)
    #print(colors[0])
    #print(colors.shape)

    colorsConv = np.zeros(colors.shape)
    if convDefs:
        for i in range(colors.shape[0]):
            colorsConv[i] = np.array(convDefs[0](*colors[i]))
    else:
        colorsConv = np.array(colors)

    if colorSpConv == 'hsv':
        colorsLinS = np.zeros([3, _len])
        # only allow wrapping of hue
        colorsLinS[0] = interpolate(colorsConv[:,0], _len, interp[0], *interp[1:], wrapPath=True)
        colorsLinS[1] = interpolate(colorsConv[:,1], _len, interp[0], *interp[1:])
        colorsLinS[2] = interpolate(colorsConv[:,2], _len, interp[0], *interp[1:])
        colorsLinS = np.transpose(colorsLinS)
    else:
        colorsLinS = np.apply_along_axis(interpolate, 0, colorsConv, _len, interp[0], *interp[1:])
    
    colorsTo = np.zeros(colorsLinS.shape)
    if convDefs:
        for i in range(colorsLinS.shape[0]):
            colorsTo[i] = np.array(convDefs[1](*colorsLinS[i]))
    else:
        colorsTo = np.array(colors)
    
    colorsTo *= 255

    return colorsTo

def draw_test_step(colorToInterp, outF):
    stripWidth = 360*stripWidthMultiplier
    interpTypes = 10
    textWidth = 250

    def draw_interpolated(colorA, colorB, yIdx, colorSpConv, interp, xStart=0):
        y_start = getYStart(yIdx)
        y_end = y_start + stripHeight

        colorsAll = getInterpolatedColors(colorA, colorB, colorSpConv, interp, stripWidth)
        for x in range(stripWidth):
            colorsTo = colorsAll[x]
            colorsTo = colorsTo.astype(int)
            colorsTo = tuple(colorsTo)

            draw.line(((x+xStart, y_start), (x+xStart, y_end)), fill=colorsTo)

        draw.rectangle([(xStart, y_start), (xStart+stripWidth, y_end)], outline='black', width=2)

        print('\n\n')

    imgW = stripWidth + textWidth + stripMargin
    imgH = (stripHeight+(2*stripMargin))*interpTypes

    img = Image.new('RGB', (imgW, imgH), ImageColor.getrgb('white'))
    draw = ImageDraw.Draw(img)

    colorA = ImageColor.getrgb(colorToInterp[0])
    colorB = ImageColor.getrgb(colorToInterp[1])

    yIdx = 0
    draw_interpolated(colorA, colorB, yIdx, 'rgb', 'lin', xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "RGB\nLin", fill='black', font_size=32)

    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', ('lin'), xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nLin", fill='black', font_size=32)

    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', ('smoothstep', 1), xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nSmoothStep_1", fill='black', font_size=32)

    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', ('smoothstep', 2), xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nSmoothStep_2", fill='black', font_size=32)

    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', ('smoothstep', 10), xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nSmoothStep_10", fill='black', font_size=32)

    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', 'quad', xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nQuad", fill='black', font_size=32)
    yIdx += 1
    draw_interpolated(colorA, colorB, yIdx, 'hsv', 'sin', xStart=textWidth)
    draw.multiline_text((0, getYStart(yIdx)), "HSV\nSine", fill='black', font_size=32)

    yIdx += 1

    imgH = (stripHeight+(2*stripMargin))*yIdx
    newImg  = Image.new('RGB', (imgW, imgH), ImageColor.getrgb('white'))
    newImg.paste(img, None)

    newImg.save(outF)

# def draw_animation(colorToInterp, outF):
#     # todo: have as args
#     imgW = 128              # pixels, width of image
#     imgH = imgW             # square for now

#     fps = 50
#     animationTime = 4                       # sec, total animation time
#     animationFms = animationTime*fps        # total animation frames

#     colorA = ImageColor.getrgb(colorToInterp[0])
#     colorB = ImageColor.getrgb(colorToInterp[1])

#     colors = getInterpolatedColors(colorA, colorB, 'hsv', 'quad', animationFms)
#     # colors = getInterpolatedColors(colorA, colorB, 'hsv', ('smoothstep', 10), animationFms)

#     gifImg = []
#     for i in range(animationFms):
#         c = colors[i]
#         c = c.astype(int)
#         c = tuple(c)
#         im = Image.new('RGB', (imgW, imgH), c)
#         gifImg.append(im)
    
#     gifImg[0].save(outF, save_all = True, append_images = gifImg[1:], optimize = False, duration = int(1000/fps), loop=True)

def draw_animation(colorToInterp, outF):
    interpTypes = 4
    gradientWidth = 128
    imageMargin = 32
    textHeight = 75

    imgW = (gradientWidth * interpTypes) + (imageMargin * (interpTypes+1))
    imgH = textHeight + gradientWidth + 10

    fps = 50
    animationTime = 6                       # sec, total animation time
    animationFms = animationTime*fps        # total animation frames

    gifImg = []
    draws = []
    for i in range(animationFms):
        im = Image.new('RGB', (imgW, imgH), 'black')
        gifImg.append(im)
        draws.append(ImageDraw.Draw(im))

    colorA = ImageColor.getrgb(colorToInterp[0])
    colorB = ImageColor.getrgb(colorToInterp[1])

    def drawColor(idx, colorsArray, text):
        for i in range(animationFms):
            c = colorsArray[i]
            c = c.astype(int)
            c = tuple(c)

            x_start = (idx*(gradientWidth+imageMargin)) + imageMargin
            y_start = textHeight

            draws[i].rectangle([(x_start, y_start), (x_start + gradientWidth, y_start + gradientWidth)], outline='white', width=2, fill=c)

            draws[i].multiline_text((x_start, 0), text, fill='white', align='left', font_size=32)
    
    colors = getInterpolatedColors(colorA, colorB, 'rgb', 'loop_lin', animationFms)
    drawColor(0, colors, "RGB\nLin")

    colors = getInterpolatedColors(colorA, colorB, 'hsv', 'loop_lin', animationFms)
    drawColor(1, colors, "HSV\nLin")

    colors = getInterpolatedColors(colorA, colorB, 'hsv', ('loop_smoothstep', 2), animationFms)
    drawColor(2, colors, "HSV\nSS2")

    colors = getInterpolatedColors(colorA, colorB, 'hsv', 'loop_sin', animationFms)
    drawColor(3, colors, "HSV\nSin")
    
    gifImg[0].save(outF, save_all = True, append_images = gifImg[1:], optimize = False, duration = int(1000/fps), loop=0)

def main():
    global draw

    # parse input args
    parser = argparse.ArgumentParser(
                    prog='Color Gradient Tester',
                    description='This programs allows testing of different gradient interpolations')

    parser.add_argument('colorA', type=str)
    parser.add_argument('colorB', type=str)
    parser.add_argument('-o', '--out', default='out', help='Ouput image file without extension. Defaults to `out`')
    parser.add_argument('-a', '--animation',  action='store_true', help='Draws animation (gif) instead of a gradient test')
    parser.add_argument('--testSrip', action='store_true', help='Generates a test strip')
    args = parser.parse_args()

    colorToInterp = args.colorA, args.colorB
    # add # as a prefix so that Pillow can recognize the color
    colorToInterp = ['#'+c for c in colorToInterp]

    if args.animation:
        args.out += '.gif'
    else:
        args.out += '.png'

    if args.testSrip:
        draw_test_step(colorToInterp, args.out)
        return
    
    if args.animation:
        draw_animation(colorToInterp, args.out)
        return


if __name__ == "__main__":
    main()