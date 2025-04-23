import sys

import numpy as np

sys.path.append('../')
from tt_loader import TabularTextLoader
from pose.pose_tools import quat2rot


class PoseTs:
    def __init__(self, ts=-1.0, T_sr=np.identity(4)):
        """
        :param ts: pose timestamp
        :param T_sr: Sensor-Reference Transformation: Xs = T_sr @ Xr
        """
        self.T_sr = T_sr
        self.ts = ts


class PoseLoader(TabularTextLoader):
    def __init__(self, path, row_format='ts,tx,ty,tz,qx,qy,qz,qw'):
        super().__init__(path)

        self.row_format = row_format
        fp = self.row_format.split(',')

        self.data_dict = dict()
        for data_row in self.data:
            ts = np.float64(data_row[fp.index('ts')])

            T = np.identity(4)
            tx = np.float32(data_row[fp.index('tx')])
            ty = np.float32(data_row[fp.index('ty')])
            tz = np.float32(data_row[fp.index('tz')])
            T[0:3, 3] = np.array([tx, ty, tz])

            qw = np.float32(data_row[fp.index('qw')])
            qx = np.float32(data_row[fp.index('qx')])
            qy = np.float32(data_row[fp.index('qy')])
            qz = np.float32(data_row[fp.index('qz')])
            T[0:3, 0:3] = quat2rot(np.array([qx, qy, qz, qw]))

            # self.data_dict[ts] = np.array(data_row[1:], dtype=np.float32)
            self.data_dict[ts] = T

        self.ts_arr = sorted(self.data_dict.keys())
        # print('%.0f' % self.ts_arr[0])

    def get_next(self):
        pose = None
        if 0 <= self.data_idx < len(self.ts_arr):
            ts = self.ts_arr[self.data_idx]
            T_sr = self.data_dict[ts]
            pose = PoseTs(ts, T_sr)
            self.data_idx += 1
        return pose

    def get_pose_ts(self, ts):
        pose_idx = np.argmin(abs(self.ts_arr - ts))
        ts = self.ts_arr[pose_idx]
        T_sr = self.data_dict[ts]
        return PoseTs(ts, T_sr)
