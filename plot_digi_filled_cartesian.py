import matplotlib.pyplot as plt
import string
import numpy as np
import matplotlib
import random
import matplotlib.pylab as pylab
from PIL import ImageOps, Image
from datetime import timedelta, datetime

params = {
         'xtick.labelsize':16,
         'ytick.labelsize':16}
pylab.rcParams.update(params)
font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)


def generate(xdim=100, xlim=(0,10), 
             colors=['silver', 'gray'], 
             smooth=1, seed=42,
             ftype='png'):
    width = 14
    height = 7
    dpi = 100
    fig, ax = plt.subplots(figsize=(width,height))
    np.random.seed(seed)
    step = (xlim[1]-xlim[0])/xdim 
    x = np.arange(xlim[0], xlim[1]+step, step)
    ys = []
    ynum = len(colors)
    for _ in range(ynum):
        y = np.random.rand(xdim+1)
        ys.append(y)
    #ax.plot(x, ys[0])
    for i in range(ynum):
        for _ in range(smooth):
            aver = (np.roll(ys[i], -1) + np.roll(ys[i], 1) + ys[i]) / 3.
            ys[i][1:-1] = aver[1:-1] + (ynum - 1 - i)
        ax.fill_between(x, ys[i], color=colors[i]) 
    ylim = (0, ynum)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    l, b, w, h = ax.get_position().bounds
    _f = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    fname = '/tmp/'+_f+'.'+ftype
    
    plt.savefig(fname)
    plt.tight_layout()
    left = int(width * dpi * l) + 1
    right = int(width * dpi * (l+w)) - 1
    bottom = int(height * (1-b) * dpi) -1
    top = int(height * (1-b-h) * dpi) + 1
    return {'fname': fname,
            'left': (left, xlim[0]), 
            'bottom': (bottom, ylim[0]), 
            'zero': (bottom, ylim[0]), #TODO define zero
            'right': (right, xlim[1]), 
            'top': (top, ylim[1])}

def get_top_boundary(pth, threshold, figdata):
    img = Image.open(pth)
    gray_image = ImageOps.grayscale(img)
    data = np.asarray(gray_image) 
    left, right = figdata['left'][0], figdata['right'][0]
    
    zero = figdata['zero'][0]
    bottom_offset = 10
    h = []
    ys = []
    for x in range(left, right):
        for y in range(zero-bottom_offset, 0, -1):
            if data[y, x] > threshold:
                h.append(zero-y+1)
                ys.append(y)
                print('FOUND')
                break
    xlim = figdata['left'][1], figdata['right'][1]
    ylim = figdata['bottom'][1], figdata['top'][1]
    bottom, top = figdata['bottom'][0], figdata['top'][0]
    
    xscale =  (xlim[1] - xlim[0]) / (right - left)
    yscale =  (ylim[1] - xlim[0]) / (bottom - top)
    xs = [(x-left) * xscale + xlim[0] for x in range(left, right)]
    ys = [(-_y+zero) * yscale + figdata['zero'][1] for _y in ys]
    return np.array(xs), np.array(ys)

def digitize_plot(figdata):
    pth = figdata['fname']
    data = get_top_boundary(pth, 130, figdata)
    data_outer = get_top_boundary(pth, 200, figdata)

    fig, ax = plt.subplots(figsize=(21,7))
    
    ax.fill_between(data_outer[0], data_outer[1], '0.8', color='gray') 
    ax.fill_between(data[0], data[1], '0.8', color='black')

    plt.tight_layout()
    plt.show()
    
figdata = generate()
digitize_plot(figdata)



