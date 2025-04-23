import os
import glob
import argparse

import numpy as np

from base_loader import DataLoader


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    mylist = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
              len(line) > 0 and line[0] != "#"]
    # cell values might be textual
    mylist = [[v for v in l] for l in mylist if len(l) > 0]
    return mylist


class TabularTextLoader(DataLoader):
    def __init__(self, path):
        super().__init__()
        self.supported_formats = ['txt', 'csv']

        self.path = None
        self.data_idx = 0
        self.file_idx = 0
        self.data = []
        self.files = []

        if self.check_file(path):
            self.path_mode = 'file'
            self.path = path
            self.data = read_file_list(path)
        elif os.path.isdir(path):
            self.path_mode = 'dir'
            self.path = path
            files = glob.glob(os.path.join(path, '*'))
            files = sorted(files)
            for file in files:
                if self.check_file(file):
                    self.files.append(file)
            if len(self.files) > 0:
                self.data = read_file_list(self.files[0])
                self.file_idx += 1

    def check_file(self, file_path):
        file_ext = os.path.splitext(file_path)[-1][1:]
        return os.path.isfile(file_path) and file_ext in self.supported_formats

    def get_data(self):
        return self.data

    @staticmethod
    def convert(data_el):
        try:
            return np.float64(data_el)
        except ValueError:
            return data_el

    def convert_v(self, data):
        if data is None:
            return []
        return [self.convert(v) for v in data]

    def convert_all(self):
        if self.data is not None:
            return np.array(self.data, dtype=np.float64)
        return self.data

    def get_cols(self, i, dtype=None, cols_slice=None):
        cols = []
        if self.data is not None:
            if cols_slice is not None:
                cols = [v[cols_slice] for v in self.data]
            else:
                cols = [v[i] for v in self.data if 0 <= i < len(v)]
        if dtype is not None:
            return np.array(cols, dtype=dtype)
        return cols

    def get_next(self):
        data = None
        if 0 <= self.data_idx < len(self.data):
            data = self.data[self.data_idx]
            self.data_idx += 1
        elif len(self.files) > 0 and 0 <= self.data_idx < len(self.files):
            self.data = read_file_list(self.files[self.file_idx])
            self.file_idx += 1
            self.data_idx = 0
            data = self.get_next()
        return self.convert_v(data)

    def swap_col(self):
        print('swap_col')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
        Testing tabular text loader
    ''')
    parser.add_argument('path', help='path of tabular dataset file or directory')
    args = parser.parse_args()

    dataset = TabularTextLoader(args.path)
    # print(dataset.get_next())
    # print(dataset.convert_all()[1])
    print(dataset.get_cols(0, dtype=np.float64)[:5])
    print(dataset.get_cols(0, dtype=np.float32, cols_slice=slice(1, 4, None))[:5])

