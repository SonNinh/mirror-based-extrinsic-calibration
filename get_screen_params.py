
import gi.repository
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk


display = Gdk.Display.get_default()
screen = display.get_default_screen()
default_screen = screen.get_default()
num = default_screen.get_number()

h_mm = default_screen.get_monitor_height_mm(num)
w_mm = default_screen.get_monitor_width_mm(num)


h_pixels = default_screen.get_height()
w_pixels = default_screen.get_width()


print(h_mm, w_mm)
print(h_pixels, w_pixels)