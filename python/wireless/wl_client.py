import time

from wl_socks import Client, MyWirelessDriver


class ControlClient(Client):
    def __init__(self, host, port, send_delay=0.01, show_time=False):
        super().__init__(host, port, show_time=show_time)
        self.should_stop = False
        # the more this delay is, the faster the remote server reacts
        self.send_delay = send_delay

    def serialize_state(self):
        # payload = MyWirelessDriver.array_to_str_v((self.v_vel, self.v_pb, self.v_sw), (2, 0, 0))
        # dummy payload
        payload = '0.0,0.0,0.0,0.0'
        return MyWirelessDriver.encode('state', payload)

    def send_control(self):
        while True:
            # serialize state
            msg_bytes = self.serialize_state()
            # send state to remote robot
            self.send_c(msg_bytes)
            if self.should_stop:
                self.send_c(MyWirelessDriver.encode('stop', '0,0'))
                break

            time.sleep(self.send_delay)


if __name__ == "__main__":

    host = '192.168.10.172'
    port = 29456
    my_client = Client(host, port)

    time.sleep(1)
    my_client.connect()
    time.sleep(1)
    my_client.send_c(b"hello")

    my_client.close()

