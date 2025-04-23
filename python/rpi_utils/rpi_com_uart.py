"""
    RPI UART Comm with other devices (microcontrollers)
    Uses pyserial (exactly like when we connect to Arduino via USB) -> refer to class `Arduino`
    (Still, wiringpi is applicable in all cases)

    refs:
        - [Raspberry Pi UART Communication using Python and C](https://www.electronicwings.com/raspberry-pi/raspberry-pi-uart-communication-using-python-and-c)
        - [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
        - [UART - Universal Asynchronous Receiver/Transmitter](https://pinout.xyz/pinout/uart)

"""

import serial
from time import sleep


def t_electronicwings():
    # UART communication on Raspberry Pi using Pyhton
    # http://www.electronicwings.com

    ser = serial.Serial ("/dev/ttyS0", 9600)    #Open port with baud rate
    while True:
        received_data = ser.read()              #read serial port
        sleep(0.03)
        data_left = ser.inWaiting()             #check for remaining byte
        received_data += ser.read(data_left)
        print (received_data)                   #print received data
        ser.write(received_data)                #transmit data serially
