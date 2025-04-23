"""
Raspberry Pi PWM and Servo generation
Most of these use GPIO RPI package (wringPi for C)
There are lots of packages for this purpose (and for working with GPIO in general)
    e.g. gpiozero

refs:
    - [PWM via DMA for the Raspberry Pi](https://pythonhosted.org/RPIO/pwm_py.html)
    - [Raspberry Pi PWM Tutorial](https://circuitdigest.com/microcontroller-projects/raspberry-pi-pwm-tutorial)
    - [Raspberry Pi PWM Generation using Python and C]
        (https://www.electronicwings.com/raspberry-pi/raspberry-pi-pwm-generation-using-python-and-c)
"""

import RPi.GPIO as GPIO
import time


class RpiPwm:
    def __init__(self, num_channels=4, freq=50):
        self.freq = freq
        self.period_us = 1000000.0 / freq
        # init 6 pwm channels (list numbers are GPIOs):
        self.gpio_pins = [32, 33, 36, 38, 40, 37]
        self.gpio_list = [12, 13, 16, 20, 21, 26]
        self.pwm_channels = []
        # pwm: for DC motors, servo: 50Hz 1ms~2ms pulse
        self.mode = 'pwm'

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        for i, gpio in enumerate(self.gpio_list):
            if i >= num_channels:
                break

            GPIO.setup(gpio, GPIO.OUT)
            p = GPIO.PWM(gpio, self.freq)
            p.start(0)
            self.pwm_channels.append(p)

    def update_duty_cycle(self, ch_id, pct):
        if ch_id < 0 or ch_id > len(self.pwm_channels):
            return
        self.pwm_channels[ch_id].ChangeDutyCycle(int(pct))

    def update_servo(self, ch_id, time_us):
        self.update_duty_cycle(ch_id, time_us * 100.0 / self.period_us)

    def update_analog(self, ch_id, trot):
        self.update_duty_cycle(ch_id, trot * 100.0 / 255.0)

    def clean(self):
        for pwm_ch in self.pwm_channels:
            pwm_ch.stop()
        GPIO.cleanup()


def t_pwm_other():
    # This also works
    gpio_num = 21   # pin 40
    GPIO.setwarnings(False)  # do not show any warnings
    GPIO.setmode(GPIO.BCM)  # we are programming the GPIO by BCM pin numbers. (PIN35 as 'GPIO19')
    GPIO.setup(gpio_num, GPIO.OUT)  # initialize GPIO19 as an output.
    p = GPIO.PWM(gpio_num, 100)  # GPIO19 as PWM output, with 100Hz frequency
    p.start(0)  # generate PWM signal with 0% duty cycle

    while True:  # execute loop forever
        for x in range(50):  # execute loop for 50 times, x being incremented from 0 to 49.
            p.ChangeDutyCycle(x)  # change duty cycle for varying the brightness of LED.
            time.sleep(0.01)  # sleep for 10m second
        time.sleep(0.5)

        for x in range(50):  # execute loop for 50 times, x being incremented from 0 to 49.
            p.ChangeDutyCycle(50 - x)  # change duty cycle for changing the brightness of LED.
            time.sleep(0.01)  # sleep for 10m second
        time.sleep(0.5)


def t_fade_led():
    # This works on most pins (12, 33, 40, 38, 32, ...)
    led_pin = 40  # PWM pin connected to LED
    GPIO.setwarnings(False)  # disable warnings
    GPIO.setmode(GPIO.BOARD)  # set pin numbering system
    GPIO.setup(led_pin, GPIO.OUT)
    pi_pwm = GPIO.PWM(led_pin, 1000)  # create PWM instance with frequency
    pi_pwm.start(0)  # start PWM of required Duty Cycle
    while True:
        for duty in range(0, 101, 1):
            pi_pwm.ChangeDutyCycle(duty)  # provide duty cycle in the range 0-100
            time.sleep(0.01)
        time.sleep(0.5)

        for duty in range(100, -1, -1):
            pi_pwm.ChangeDutyCycle(duty)
            time.sleep(0.01)
        time.sleep(0.5)


def t_fade_rpi_pwm():
    pwm_generator = RpiPwm()

    while True:
        for duty in range(0, 101, 1):
            pwm_generator.update_duty_cycle(0, duty)
            time.sleep(0.01)
        time.sleep(0.5)

        for duty in range(100, -1, -1):
            pwm_generator.update_duty_cycle(0, duty)
            time.sleep(0.01)
        time.sleep(0.5)


def t_rpi_pwm_constant():
    pwm_generator = RpiPwm()
    pwm_generator.update_duty_cycle(0, 5)
    time.sleep(10)
    pwm_generator.update_duty_cycle(0, 0)


if __name__ == "__main__":

    # t_pwm_other()
    # t_fade_led()
    # t_fade_rpi_pwm()
    t_rpi_pwm_constant()
	
	
	
	
