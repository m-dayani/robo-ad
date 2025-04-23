"""
    Working with ultrasonic sensor

    refs:
        - [Raspberry Pi Distance Sensor using the HC-SR04](https://pimylifeup.com/raspberry-pi-distance-sensor/)
        - [Using a Raspberry Pi distance sensor (ultrasonic sensor HC-SR04)](https://tutorials-raspberrypi.com/raspberry-pi-ultrasonic-sensor-hc-sr04/)
        - [Using an ultrasonic distance sensor](https://projects.raspberrypi.org/en/projects/null/12)
        - [Lesson 23: Ultrasonic Sensor Module (HC-SR04)](https://docs.sunfounder.com/projects/umsk/en/latest/05_raspberry_pi/pi_lesson23_ultrasonic.html)
"""
import ctypes
import inspect
import threading

import RPi.GPIO as GPIO
import time
from gpiozero import DistanceSensor


def t_ultrasonic_sensor():
    try:
        GPIO.setmode(GPIO.BOARD)

        PIN_TRIGGER = 7
        PIN_ECHO = 11

        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        GPIO.setup(PIN_ECHO, GPIO.IN)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        print("Waiting for sensor to settle")

        time.sleep(2)

        print("Calculating distance")

        GPIO.output(PIN_TRIGGER, GPIO.HIGH)

        time.sleep(0.00001)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        pulse_end_time = pulse_start_time = 0
        while GPIO.input(PIN_ECHO)==0:
            pulse_start_time = time.time()
        while GPIO.input(PIN_ECHO)==1:
            pulse_end_time = time.time()

        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        print("Distance:", distance, "cm")

    finally:
        GPIO.cleanup()


def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising an exception in the thread from
       another thread.
    '''
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL: this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.is_alive(): # Note: self.isAlive() on older version of Python
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do: self.ident

        raise AssertionError("could not determine the thread's id")

    def raise_exc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raise_exc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raise_exc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL: this function is executed in the context of the
        caller thread, to raise an exception in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )


class UltraSonic(object):

    def __init__(self, pin_trig=18, pin_echo=24, use_sensor=True, loop_delay=0.01):
        # GPIO Mode (BOARD / BCM)
        GPIO.setmode(GPIO.BCM)

        # set GPIO Pins
        self.GPIO_TRIGGER = pin_trig
        self.GPIO_ECHO = pin_echo

        # set GPIO direction (IN / OUT)
        GPIO.setup(self.GPIO_TRIGGER, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO, GPIO.IN)

        self.last_dist = -1
        self.should_stop = False
        self.stop_lock = threading.Lock()
        self.run_thread = None
        self.dist_lock = threading.Lock()

        self.loop_delay = loop_delay
        self.dist_available = use_sensor
        # self.test_dist_available()

        if self.dist_available:
            print('Ultrasonic is available')
        else:
            print('Ultrasonic is NOT available')

    def test_dist_available(self):
        # If no real sensor is attached to the pins,
        # you'll have a really hard time stopping the distance() method!
        thread_dist = ThreadWithExc(target=self.distance)
        thread_dist.start()
        self.dist_available = self.dist_lock.acquire(timeout=5.0)
        try:
            thread_dist.raise_exc(RuntimeError)
        except RuntimeError:
            pass

    def distance(self):
        distance = -1
        try:
            self.dist_lock.acquire()

            # set Trigger to HIGH
            GPIO.output(self.GPIO_TRIGGER, True)

            # set Trigger after 0.01ms to LOW
            time.sleep(0.00001)
            GPIO.output(self.GPIO_TRIGGER, False)

            StartTime = time.time()
            StopTime = time.time()

            # save StartTime
            while GPIO.input(self.GPIO_ECHO) == 0:
                StartTime = time.time()

            # save time of arrival
            while GPIO.input(self.GPIO_ECHO) == 1:
                StopTime = time.time()

            # time difference between start and arrival
            TimeElapsed = StopTime - StartTime
            # multiply with the sonic speed (34300 cm/s)
            # and divide by 2, because there and back
            distance = (TimeElapsed * 34300) / 2

            self.dist_lock.release()

        except RuntimeError:
            print('UltraSonic: Error computing distance')

        return distance

    def run(self):
        self.stop(False)
        while not self.is_stopped():
            if self.dist_available:
                self.last_dist = self.distance()
            time.sleep(self.loop_delay)

    def run_in_background(self):
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()
        print('UltraSonic: Started in background')

    def is_stopped(self):
        self.stop_lock.acquire()
        flag = self.should_stop
        self.stop_lock.release()
        return flag

    def stop(self, flag):
        self.stop_lock.acquire()
        self.should_stop = flag
        self.stop_lock.release()

    def get_distance(self):
        return self.last_dist

    def clean(self):
        if self.run_thread is not None:
            self.stop(True)
            self.dist_lock.acquire(timeout=5.0)
            self.run_thread.join(5)
            self.run_thread = None
            GPIO.cleanup()
            self.dist_lock.release()
            print('Ultrasonic: Finished Processing Distance in BG')


def t_gpiozero():

    # Initialize the DistanceSensor using GPIO Zero library
    # Trigger pin is connected to GPIO 17, Echo pin to GPIO 27
    sensor = DistanceSensor(echo=27, trigger=17)

    try:
        # Main loop to continuously measure and report distance
        while True:
            dis = sensor.distance * 100  # Measure distance and convert from meters to centimeters
            print('Distance: {:.2f} cm'.format(dis))  # Print the distance with two decimal precision
            time.sleep(0.3)  # Wait for 0.3 seconds before the next measurement

    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (Ctrl+C) to gracefully exit the loop
        pass


if __name__ == '__main__':

    sensor = UltraSonic()

    try:
        while True:
            dist = sensor.distance()    # distance()
            print("Measured Distance = %.1f cm" % dist)
            time.sleep(1)

        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
