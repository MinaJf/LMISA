import os
import glob
import time
import shutil
from scipy.ndimage import zoom
import random
import numpy as np
from utils import data_loader as L
from utils import util as U
from scipy import ndimage
import cv2
class DataProvider:
    """Callable data provider. 

    Parameter
    ----------
    data: str, list or dict
        str: Path to search the file.
        list: A list of all filenames. 
        dict: A dict for all data.
    data_suffix: list or tuple
        a list of suffix, the first value should be the suffix of input images.
    processor: obj: Processor, optional
        A processor for precess images.
    is_pre_load: bool, optional
        True: load all the files into memory. False: load files into memory only when others call it.
    is_shuffle: bool, optional
        Random shuffle data or not after taking out all data each round.

    Attributes
    ----------
    _data_suffix: str
        Suffix for input images and their corresponding key in the output.
    _other_suffixes: list of tuple, optional
        Suffix for other images and their corresponding keys in the output.
    _is_shuffle: bool
        Random shuffle data or not after taking out all data each round.
    _processor: obj: Processor
        A processor for precess images.
    _file_list: list:str
        A list of all filenames. 
    _all_data: dict
        Dict of ndarray that corresponds to the data when preloaded data is selected.
    _cur_i: int
        Current index for cycle.
    
    """
    def __init__(self,
                data,
                data_suffix ,
                processor=None,
                temp_dir=None,
                is_pre_load=False,
                is_shuffle=False,
                is_aug=False):
        assert len(data_suffix) > 0, 'Empty suffix!'
        self._org_suffix = data_suffix[0]
        self._other_suffix = data_suffix[1:]
        self._is_shuffle = is_shuffle
        self._temp_dir = temp_dir
        self._is_aug = is_aug
        self._processor = processor
        
        self._file_list = None
        self._all_data = None
        if type(data) is str:
            self._file_list = glob.glob(data)
        elif type(data) is list or type(data) is np.ndarray:
            self._file_list = data
        elif type(data) is dict:
            self._all_data = data
        else:
            raise ValueError('Only accept one of (search_path, file_list, data_dict).')

        self._cur_i = 0
        if self._file_list is not None:
            if is_pre_load:
                self._all_data = self._load_data(len(self._file_list))
            elif self._temp_dir is not None:
                self._build_temp_folder()         

    def __call__(self, n):
        """Require images.

        Parameters
        ----------
        n: int
            The number of images required.

        Returns
        -------
        dict
            A dictionary of ndarray data:
                {
                    'data_suffix':      ndarray,
                    'other_suffix 1':   ndarray,
                    'other_suffix 1':   ndarray,
                    ...
                }
            The shape of ndarray will be (n, x, y, ..., c).
                n is the number of data, which caller asked.
                x, y, ... is the size of data.
                c is the number of channels (for label is the numebr of classes).
        """
        data_dict = {}
        if self._all_data is not None:
            idx_list = np.array(range(self._cur_i, self._cur_i+n)) % len(self._file_list)
            for key in self._all_data:
                data_dict.update({key: self._all_data[key][idx_list]})
            self._next_idx(n)
        elif self._temp_dir is not None:
            data_dict.update(self._load_temp_file(n))
        else:
            data_dict.update(self._load_data(n))  
        return data_dict 

    @property
    def size(self):
        return len(self._file_list) if self._file_list is not None else len(self._all_data)

    def _load_data(self, n):
        """Load and process data one by one

        Parameters
        ----------
        n: int
            The number of images loaded.

        Returns
        -------
        dict
            A dictionary of ndarray data:
                {
                    'data_suffix':      ndarray,
                    'other_suffix 1':   ndarray,
                    'other_suffix 1':   ndarray,
                    ...
                }
            The shape of ndarray will be (n, x, y, ..., c).
                n is the number of data, which caller asked.
                x, y, ... is the size of data.
                c is the number of channels (for label is the numebr of classes).
        """
        data_dict = {}
        for _ in range(n):
            sub_data_dict = {}
            x_name = self._file_list[self._cur_i]
            sub_data_dict.update({self._org_suffix: L.load_file(x_name)})

            for o_suffix in self._other_suffix:
                o_name = x_name.replace(self._org_suffix, o_suffix)
                sub_data_dict.update({o_suffix: L.load_file(o_name)})

            # augmentation
            if self._is_aug:


                shift_deg1 = random.randint(-10, 10)
                shift_deg2 = random.randint(-5, 5)
                shift_deg3 = random.randint(-5, 5)

                sc_value_z = random.uniform(0.75, 1.20)


                AXES = [ (0, 1), (1, 2), (0, 2)]
                # AXES = [(0, 1)]
                a = random.choice(AXES)

                if (a==(1, 2)):
                    rotate_deg = random.randint(-25, 25)
                else:
                    rotate_deg = random.randint(-15, 15)

                sub_data_dict[self._org_suffix] = self.rotation(sub_data_dict[self._org_suffix] , 'img', rotate_deg, shift_deg1, shift_deg2, shift_deg3, a, sc_value_z)
                if len(self._other_suffix) > 0:
                    sub_data_dict[self._other_suffix[0]] = self.rotation(sub_data_dict[self._other_suffix[0]], 'msk', rotate_deg,  shift_deg1, shift_deg2, shift_deg3, a, sc_value_z)
            # process
            if self._processor is not None:
                sub_data_dict = self._processor.pre_process(sub_data_dict)

            # if self._is_aug and len(self._other_suffix) > 0:
            #     sub_data_dict[self._other_suffix[0]] = self.rotation(sub_data_dict[self._other_suffix[0]], 'msk', rotate_deg, shift_deg, translation_matrix)

            data_dict = U.dict_append(data_dict, sub_data_dict) 
            self._next_idx()
        U.dict_list2arr(data_dict)
        return data_dict

    def rotation(self, x, flag, rotate_deg, shift_deg1, shift_deg2, shift_deg3, a, sc_value_z):
        # np.random.seed(seed)


        if (flag == 'img'):
            _x = ndimage.rotate(x, rotate_deg, a, reshape=False, prefilter=True, order=1)
        if (flag == 'msk'):
            # _, _, _, channels = x.shape
            # _x = [ndimage.rotate(x[..., c], rotate_deg, a, reshape=False, prefilter=True, order=0)
            #       for c in range(channels)]
            # _x = np.stack(_x, axis=-1)
            _x = ndimage.rotate(x, rotate_deg, a, reshape=False, prefilter=True, order=0, mode='nearest')

        if (flag == 'img'):
           # _x = ndimage.shift(_x, [shift_deg1, shift_deg2, shift_deg3], order=1)
           _x = ndimage.shift(_x, [shift_deg1, shift_deg2, shift_deg3], order=1)
        if (flag == 'msk'):
        #    # _, _, _, channels = _x.shape
        #    # _x = [ndimage.shift(_x[..., c], shift_deg, order=0, mode='nearest')
        #    #       for c in range(channels)]
        #    # _x = np.stack(_x, axis=-1)
        #    # _x = np.round(_x)
        #
        #    # _x[0:1,:, :] = 0
        #    # _x[125:127, :, :] = 0
        #    # _x[:,0:1,:] = 0
        #    # _x[:,62:63, :] = 0
        #    _x = ndimage.shift(_x, [shift_deg1, shift_deg2, shift_deg3], order=0)
             _x = ndimage.shift(_x, [shift_deg1, shift_deg2, shift_deg3], order=0)

        # if (flag == 'img'):
        #    _x = ndimage.zoom(_x, 1.5, order=3)
        # if (flag == 'msk'):
        #     _x = ndimage.zoom(_x, 1.5, order=0)
        _x = self.clipped_zoom(_x,sc_value_z, flag)
        # _x = self.clipped_zoom(_x, sc_value_zin, flag)
        return _x

    def clipped_zoom(self, img, zoom_factor, flag ,**kwargs):

        h, w = img.shape[:2]

        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        # Zooming out
        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            if (flag == 'msk'):
                out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, order=0, **kwargs)
            if (flag == 'img'):
                out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, order=1, **kwargs)

        # Zooming in
        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.ceil(h / zoom_factor))
            zw = int(np.ceil(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            if (flag == 'msk'):
                out = zoom(img[top:top + zh, left:left + zw], zoom_tuple,order=0, **kwargs)
            if (flag == 'img'):
                out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, order=1, **kwargs)
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top + h, trim_left:trim_left + w]

        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out


    def _build_temp_folder(self):
        self._temp_dir += '/temp' + time.strftime('%Y%m%d-%H%M%S-') + str(time.time()).split('.')[-1]
        print('Building temp folder \'{}\' ...'.format(self._temp_dir))
        # temp_dir = tempfile.TemporaryDirectory()
        
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        os.makedirs(self._temp_dir)
        new_filelist = []
        for i in range(len(self._file_list)):
            data_dict = self._load_data(1)
            temp_filepath = '{}/temp_dict_{}.npy'.format(self._temp_dir, i)
            new_filelist.append(temp_filepath)
            np.save(temp_filepath, data_dict)
        self._file_list = new_filelist
        print('Processed temp files were saved.')
    
    def _load_temp_file(self, n):
        assert self._temp_dir is not None, 'Temp dir is None'
        assert os.path.exists(self._temp_dir), 'Can\'t find temp directory \'{}\''.format(self._temp_dir)
        data_dict = {}
        for _ in range(n):
            temp_filename = self._file_list[self._cur_i]
            sub_data_dict = np.load(temp_filename, allow_pickle='TRUE').item()
            self._next_idx()
            data_dict = U.dict_concat(data_dict, sub_data_dict) 
        return data_dict
            
    def _next_idx(self, n=1):
        """Cycle index.
        Parameters
        ----------
        n: int, optional
            The value needs to be added。
        """
        self._cur_i += n
        if self._cur_i >= len(self._file_list):
            self._cur_i = self._cur_i % len(self._file_list)
            if self._is_shuffle:
                shuffle_idx = np.random.permutation(len(self._file_list))
                if self._file_list is not None:
                    self._file_list = [self._file_list[i] for i in shuffle_idx]
                if self._all_data is not None:
                    for key in self._all_data:
                        self._all_data[key] = self._all_data[key][shuffle_idx]