import numpy as np
import sys

from utils import util as U
from utils import data_loader as L
from core.data_provider import DataProvider


class IDDataProvider(DataProvider):
    def __init__(self,
                 data,
                 data_suffix,
                 processor=None,
                 temp_dir=None,
                 is_pre_load=False,
                 is_shuffle=False,
                 id_suffix=None):
        super().__init__(data, data_suffix, processor, temp_dir, is_pre_load, is_shuffle)
        self._id_suffix = id_suffix
        self._lab_suffix = data_suffix[1]

    def _load_data(self, n):
        data_dict = {}
        for _ in range(n):
            sub_data_dict = {}
            x_name = self._file_list[self._cur_i]
            sub_data_dict.update({self._org_suffix: L.load_file(x_name, 0)})

            x1_name = self._file_list[np.random.randint(0, len(self._file_list) - 1)]
            sub_data_dict.update({self._org_suffix: L.load_file(x1_name, 0)})

            for o_suffix in self._other_suffix:
                o_name = x_name.replace(self._org_suffix, o_suffix)
                sub_data_dict.update({o_suffix: L.load_file(o_name, 1)})

                o1_name = x1_name.replace(self._org_suffix, o_suffix)
                sub_data_dict.update({o_suffix: L.load_file(o1_name, 1)})

                # process
            if self._processor is not None:
                sub_data_dict = self._processor.pre_process(sub_data_dict)

            data_dict = U.dict_append(data_dict, sub_data_dict)
            self._next_idx()
        U.dict_list2arr(data_dict)
        return data_dict
