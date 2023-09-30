#!/usr/bin/env python3
import json
import logging
import time

from cereal import messaging
from openpilot.common.realtime import Ratekeeper

TIME_GAP_THRESHOLD = 0.5
last_control_send_time = time.monotonic()
logger = logging.getLogger("pc")
logging.basicConfig(level=logging.INFO)


def send_control_message(pm, x, y, source):
  global last_control_send_time
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [x, y]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)
  logger.info(f"bodycontrol|{source} (x, y): ({x}, {y})")
  last_control_send_time = time.monotonic()

lablemap = {0: "straight", 1: "left", 2: "right"}
commandmap = {0: [-1, 0], 1: [0, 1], 2: [0, -1]}

def main():
  rk = Ratekeeper(20.0)
  pm = messaging.PubMaster(['testJoystick'])
  sm = messaging.SubMaster(['customReservedRawData0', 'customReservedRawData1'])

  while True:
    sm.update(0)

    if sm.updated['customReservedRawData0']:
      controls = json.loads(sm['customReservedRawData0'].decode())
      send_control_message(pm, controls['x'], controls['y'], 'wasd')
    elif sm.updated['customReservedRawData1']:
      # ToDo: do something with the yolo outputs
      label = sm['customReservedRawData1'].decode()
      print(f'Command -> {lablemap[label]}')
      control = commandmap[label]
      send_control_message(pm, control[0], control[1], 'wasd')

    else:
      now = time.monotonic()
      if now > last_control_send_time + TIME_GAP_THRESHOLD:
        send_control_message(pm, 0.0, 0.0, 'dummy')

    rk.keep_time()


if __name__ == "__main__":
  main()
