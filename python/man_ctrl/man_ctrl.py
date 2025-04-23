# Note: pynput needs screen, keyboard needs a real keyboard attached to RPI
#       sshkeyboard can operate over Wi-Fi but have limitations

import numpy as np


class ManualControl:
    def __init__(self):
        # sensitivity
        self.MAX_SPEED = 30

        self.stop_man_ctrl = False
        self.man_ctrl_running = False

        # We have 4 continuous values, 4 sticky switches, and 4 push-buttons
        # v_velocity: [+f/-b, -l/+r, +u/-d(trot), +cw/-ccw]
        self.v_vel = np.zeros(4, dtype=np.float32)
        self.v_sw = np.zeros(4, dtype=np.bool_)
        self.v_pb = np.zeros(4, dtype=np.bool_)

    def run(self):
        while not self.stop_man_ctrl:
            print('Base controller running')
            self.stop_man_ctrl = True

    def on_press(self, key):
        print('Base controller on_press: ' + str(key))
        print(self.v_vel)

    def on_release(self, key):
        print('Base controller on_release: ' + str(key))
        print(self.v_vel)

    def update(self):
        print(self.v_vel)
        return False

    def clean(self):
        print('Base controller clean')
        self.v_vel = np.zeros(4, dtype=np.float32)
        self.v_sw = np.zeros(4, dtype=np.bool_)
        self.v_pb = np.zeros(4, dtype=np.bool_)


def t_man_ctrl(controller):
    controller.run()


if __name__ == "__main__":

    my_controller = ManualControl()
    t_man_ctrl(my_controller)
    my_controller.clean()

