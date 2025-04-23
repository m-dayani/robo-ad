import time

from wl_socks import Server, MyWirelessDriver


class ControlServer(Server):
    def __init__(self, host, port, show_time=False):
        super().__init__(host, port, show_time=show_time)

    def process_cmd(self, cmd, data):
        if cmd == 'state':
            # update control state
            # state_vec = np.float32(data.split(','))
            print('remote server: ' + str(data))
        elif cmd == 'stop':
            return False

    def server_trans(self):
        res, data = self.receive(self.client_socket, -1, self.MSG_LEN)
        if not res:
            return res
        if len(self.raw_data) > 0:
            data = self.raw_data + data
        msg_arr, residual = MyWirelessDriver.decode(data)
        # print(msg_arr)
        for msg in msg_arr:
            label = ''
            payload = ''
            if len(msg) >= 1:
                label = msg[0]
            if len(msg) == 2:
                payload = msg[1]

            if len(label) > 0:
                res = self.process_cmd(label, payload)
                if not res:
                    return res

        if residual is not None:
            self.raw_data += bytes(residual, 'utf-8')
            # print('message residual not empty: ' + residual)
        else:
            self.raw_data = b''

        return True


if __name__ == "__main__":

    host = '192.168.10.172'
    port = 29456
    my_server = Server(host, port)

    my_server.run()
    time.sleep(1)
    print(MyWirelessDriver.decode_str(my_server.raw_data))

    my_server.set_started(False)
    my_server.close()
