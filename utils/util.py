import numpy as np
import math
def dict_append(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        for key in new_dict:
            old_dict[key] = []

    for key in old_dict:
        assert key in new_dict, 'No key "{}" in old dict!'.format(key)
        old_dict[key].append(new_dict[key])

    return old_dict

def dict_concat(old_dict, new_dict, axis=0):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        old_dict = new_dict
    else:
        for key in old_dict:
            assert key in new_dict, 'No key "{}" in old dict!'.format(key)
            old_v = old_dict[key]
            new_v = new_dict[key]
            old_v = [old_v] if np.ndim(old_v) == 0 else old_v
            new_v = [new_v] if np.ndim(new_v) == 0 else new_v
            old_dict[key] = np.concatenate((old_v, new_v), axis)

    return old_dict


def dict_add(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None:
        old_dict = new_dict
    else:
        for key in new_dict:
            old_dict[key] += new_dict[key]
    return old_dict

def dict_list2arr(d):
    for key in d:
        d[key] = np.array(d[key])

def dict_to_str(evaluation_dict, axis=0):
    if evaluation_dict is None or not evaluation_dict:
        return ''
    o_s = ''
    for key in evaluation_dict:
        value = np.array(evaluation_dict.get(key))
        #value[value==0] = math.nan
        if value.size >= 2:
            mean = np.nanmean(value, axis)  # [1:]
        else:
            mean = value
        if type(mean) in [int, float, np.float32, np.float64]:
            mean = [mean]
        if np.ndim(mean) > 1:
            continue
        if value.size >= 2:
            mean = ['%.4f' % m for m in mean]
        else:
            mean = ['%.4f' % mean]
        o_s += '%s: ' % key
        for s in mean:
            o_s += '%s ' % s
        o_s += '  '
    return o_s


def recale_array(array, nmin=0, nmax=4, tmin=0, tmax=255, dtype=np.uint8):
    array = np.array(array)
    if nmin is None:
        nmin = np.min(array)
    array = array - nmin
    if nmax is None:
        nmax = np.max(array) + 1e-9
    array = array / nmax
    array = (array * (tmax - tmin)) - tmin
    return array.astype(dtype)

def gray2rgb(img):
    return np.stack((img,)*3, axis=-1)

def combine_2d_imgs_from_tensor(img_list):
    imgs = []
    combined = None
    for im in img_list:
        assert len(im.shape) == 3 or len(im.shape) == 4 and im.shape[-1] in [1, 3], \
        'Only accept gray or rgb 2d images with shape [n, x, y] or  [n, x, y, c], where c = 1 (gray) or 3 (rgb).'
        if im.shape[-1] != 3:
            if len(im.shape) == 4:
                im = im[..., 0]
            im = gray2rgb(im)
            
        im = recale_array(im)
        im = im.reshape(-1, im.shape[-2], im.shape[-1])
        imgs.append(im)
    combined = np.concatenate(imgs, 1)
    return combined

