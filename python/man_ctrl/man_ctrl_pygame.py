#!/usr/bin/env python
"""
    No need for python-uinput (and superuser privileges)
"""

import time
import os

import pygame
from pygame.locals import *

from man_ctrl import ManualControl


class MyJoystick(ManualControl):
    def __init__(self):
        super().__init__()

        pygame.init()
        BLACK = (0,0,0)
        WIDTH = 335
        HEIGHT = 469
        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.windowSurface.fill(BLACK)
        pygame.display.set_caption('Virtual RC Joystick')

        # Fill background
        background = pygame.Surface(self.windowSurface.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Load image
        self.dir = os.path.dirname(__file__)
        filename = os.path.join(self.dir, 'assets/sticks_with_buttons.png')
        img = pygame.image.load(filename)

        self.windowSurface.blit(img,(0,0))
        pygame.display.flip()

        self.sticks = []
        self.switches = []

    def run(self):
        # create sticks
        pitch_stick = StickState('Pitch', K_UP, K_DOWN, self.dir)
        pitch_stick.set_display(328, 198, False)
        self.sticks.append(pitch_stick)
        roll_stick = StickState('Roll', K_RIGHT, K_LEFT, self.dir)
        roll_stick.set_display(21, 39, True)
        self.sticks.append(roll_stick)
        thr_stick = StickState('Throttle', K_w, K_s, self.dir, False)
        thr_stick.set_display(328, 95, False)
        self.sticks.append(thr_stick)
        rud_stick = StickState('Yaw', K_d, K_a, self.dir)
        rud_stick.set_display(360, 39, True)
        self.sticks.append(rud_stick)

        # create switches
        switch_1 = SwitchState('1', K_q, self.dir)
        switch_1.set_display(419, 79)
        self.switches.append(switch_1)
        switch_2 = SwitchState('2', K_1, self.dir, False)
        switch_2.set_display(419, 131)
        self.switches.append(switch_2)
        switch_3 = SwitchState('3', K_2, self.dir, False)
        switch_3.set_display(419, 183)
        self.switches.append(switch_3)
        switch_4 = SwitchState('4', K_3, self.dir, False)
        switch_4.set_display(419, 236)
        self.switches.append(switch_4)

        while True:
            # event handling loop
            should_break = False
            for event in pygame.event.get():
                for stick in self.sticks:
                    stick.update_event(event)
                for switch in self.switches:
                    switch.update_event(event, self.windowSurface)
                should_break = event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE)
            for stick in self.sticks:
                stick.update_stick(self.windowSurface)
            self.stop_man_ctrl = should_break
            self.update()
            if should_break:
                break
            time.sleep(0.0005)

    def update(self):
        # update sticks
        for i, stick in enumerate(self.sticks):
            if i >= 4:
                break
            self.v_vel[i] = stick.val
        cnt_sw = 0
        cnt_pb = 0

        # update switches
        for switch in self.switches:
            if switch.spring_back:
                # push-button mode:
                self.v_pb[cnt_pb] = switch.active
                cnt_pb += 1
            else:
                # switch mode:
                self.v_sw[cnt_sw] = switch.active
                cnt_sw += 1

class StickState(object):
    def __init__(self, name, key_up, key_down, dir_name, spring_back=True, incr_val=0.2):
        self.name = name                            # The name of the stick
        # self.stick = stick                          # The stick on the joystick that this stick maps to
        self.key_up = key_up                        # The key on the keyboard that maps to this stick increment
        self.key_down = key_down                    # The key on the keyboard that maps to this stick decrement
        self.spring_back = spring_back              # Does the stick spring back to center on release?
        self.incr_val = incr_val                    # The increment on keypress
        self.min_val = 0.0                          # Minimum stick value
        self.max_val = 255.0                        # Maximum stick value
        self.active_up = False                      # True if up key is held pressed
        self.active_down = False                    # True if down key is held pressed
        if self.spring_back:
            self.zero = 127.0
        else:
            self.zero = 0.0
        self.val = self.zero                        # Stick value at initialization at zero position
        self.emit_val = int(self.val)
        self.display_ready = False                  # Whether optional display params have been set
        self.display_height = 0                     # Height on the display screen
        self.display_width = 0                      # Width on the display screen
        self.display_hor = True                     # Whether the display bar is horizontal, else vertical
        self.display_bar_g = []
        self.display_bar_b = []
        self.dir = dir_name

    def keypress_up(self):
        self.active_up = True
        if (self.val + self.incr_val) <= self.max_val:
            self.val = self.val + self.incr_val
        else:
            # Saturated
            self.val = self.max_val

    def keypress_down(self):
        self.active_down = True
        if (self.val - self.incr_val) >= self.min_val:
            self.val = self.val - self.incr_val
        else:
            # Saturated
            self.val = self.min_val

    def release_stick(self):
        if not self.spring_back:
            pass
        else:
            if self.val > self.zero:
                self.val = self.val - self.incr_val*0.2
            elif self.val < self.zero:
                self.val = self.val + self.incr_val*0.2
            else:
                self.val = self.zero

    def emit(self, device):
        # emit efficiently
        if abs(int(round(self.val)) - int(self.emit_val)) > 0.001:
            self.emit_val = int(round(self.val))
            # device.emit(self.stick, int(self.emit_val), syn=False)
            if self.display_ready:
                self.display(device)

    def set_display(self, offset_height, offset_width, horizontal):
        self.display_height = offset_height
        self.display_width = offset_width
        self.display_hor = horizontal
        if horizontal:
            filename = os.path.join(self.dir, 'assets/hg.png')
            self.display_bar_g = pygame.image.load(filename)
            filename = os.path.join(self.dir, 'assets/hb.png')
            self.display_bar_b = pygame.image.load(filename)
        else:
            filename = os.path.join(self.dir, 'assets/vg.png')
            self.display_bar_g = pygame.image.load(filename)
            filename = os.path.join(self.dir, 'assets/vb.png')
            self.display_bar_b = pygame.image.load(filename)
        self.display_ready = True

    def display(self, windowSurface):
        if not self.display_ready:
            pass
        else:
            # Fill the entire bar
            for i in range(256):
                if i <= self.emit_val:
                    # Fill green
                    if self.display_hor:
                        windowSurface.blit(self.display_bar_g,(self.display_width + i, self.display_height))
                    else:
                        windowSurface.blit(self.display_bar_g,(self.display_width, self.display_height - i))
                else:
                    # Fill grey
                    if self.display_hor:
                        windowSurface.blit(self.display_bar_b,(self.display_width + i, self.display_height))
                    else:
                        windowSurface.blit(self.display_bar_b,(self.display_width, self.display_height - i))
            # Render it
            pygame.display.flip()

    def update_event(self, event):
        if event.type == KEYUP:
            if event.key == self.key_up:
                self.active_up = False
            elif event.key == self.key_down:
                self.active_down = False
        elif event.type == KEYDOWN:
            if event.key == self.key_up:
                self.active = True
                self.keypress_up()
            elif event.key == self.key_down:
                self.active = True
                self.keypress_down()

    def update_stick(self, device):
        if self.active_up:
            self.keypress_up()
            self.emit(device)
        elif self.active_down:
            self.keypress_down()
            self.emit(device)
        else:
            self.release_stick()
            self.emit(device)

class SwitchState(object):
    def __init__(self, name, key, dir_name, spring_back=True):
        self.name = name                            # The name of the button
        # self.switch = switch                        # The switch on the joystick that this switch maps to
        self.key = key                              # The key on the keyboard that maps to this switch
        self.spring_back = spring_back              # Does the 'switch' switch back to off-state on release?
        self.active = False                         # True if key is held pressed
        self.active_previous = False                # Holds the previous active value
        self.display_ready = False                  # Whether optional display params have been set
        self.display_height = 0                     # Height on the display screen
        self.display_width = 0                      # Width on the display screen
        self.display_clicked = []
        self.display_released = []
        self.dir = dir_name

    def keypress(self):
        self.active_previous = self.active
        if self.spring_back:
            self.active = True                      # self.active toggles automatically on key release
        else:
            self.active = not self.active           # toggle self.active on keypress

    def release_switch(self):
        if not self.spring_back:
            pass
        else:
            self.active_previous = self.active
            self.active = False

    def emit(self, device):
        # emit efficiently
        if self.active_previous != self.active:
            # device.emit(self.switch, int(self.active))
            self.active_previous = self.active
            if self.display_ready:
                self.display(device)

    def set_display(self, offset_height, offset_width):
        self.display_height = offset_height
        self.display_width = offset_width
        filename = os.path.join(self.dir, 'assets/clicked.png')
        self.display_clicked = pygame.image.load(filename)
        filename = os.path.join(self.dir, 'assets/released.png')
        self.display_released = pygame.image.load(filename)
        self.display_ready = True

    def display(self, windowSurface):
        if not self.display_ready:
            pass
        else:
            if self.active:
                windowSurface.blit(self.display_clicked,(self.display_width, self.display_height))
            else:
                windowSurface.blit(self.display_released,(self.display_width, self.display_height))
            # Render it
            pygame.display.flip()

    def update_event(self, event, device):
        if event.type == KEYUP:
            if event.key == self.key:
                self.release_switch()
                self.emit(device)
        elif event.type == KEYDOWN:
            if event.key == self.key:
                self.keypress()
                self.emit(device)


if __name__ == "__main__":
    # main()
    joystick = MyJoystick()
    joystick.run()