import time
from core import Engine
from plugins.dynamics import Compressor
from plugins.basic_nodes import SineOscillator
from plugins.audio_devices import AudioDeviceOutput

engine = Engine()
engine.start()
time.sleep(0.5)

osc = SineOscillator()
out = AudioDeviceOutput()
c = Compressor()
c.id = "c1"
osc.id = "osc1"
out.id = "out1"

engine.push_command(("add", osc, osc.id, (0,0), {}))
engine.push_command(("add", out, out.id, (0,0), {}))
engine.push_command(("add", c, c.id, (0,0), {}))
time.sleep(0.5)

engine.push_command(("conn", osc.id, "out", c.id, "in"))
engine.push_command(("conn", c.id, "out", out.id, "in"))
time.sleep(1.0)

print("Deleting...")
engine.push_command(("del", c.id))
time.sleep(1.0)
engine.stop()
print("Success!")
