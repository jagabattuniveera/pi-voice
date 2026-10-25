#!/usr/bin/env python3
"""
Minimal ILI9488 driver for Raspberry Pi 5 (userspace: spidev + gpiozero + Pillow).

The ILI9488 only accepts 18-bit colour over SPI, so every pixel is sent as
3 bytes (R, G, B) - exactly what PIL's img.tobytes() produces for an RGB image.

Wiring (SPI0):                       Pi header pin
  VCC   -> 5V (or 3.3V, see module)  1
  GND   -> GND                       6
  CS    -> GPIO8  (CE0)              24
  RESET -> GPIO25                    22
  DC/RS -> GPIO24                    18
  SDI   -> GPIO10 (MOSI)             19
  SCK   -> GPIO11 (SCLK)             23
  LED   -> 3.3V (or GPIO18 for PWM)  17
  SDO   -> leave unconnected (display never needs to be read)

Run this file directly for a test pattern + live clock:
  python ili9488.py
"""
import time

import spidev
from gpiozero import OutputDevice, PWMOutputDevice
from PIL import Image, ImageDraw, ImageFont

# MADCTL values for rotation 0..3 (0x08 = BGR order bit)
_MADCTL = [0x48, 0x28, 0x88, 0xE8]


class ILI9488:
    def __init__(self, dc=24, rst=25, bl=None, bus=0, device=0,
                 speed_hz=32_000_000, rotation=1, invert=False, bgr=True):
        self.dc = OutputDevice(dc)
        self.rst = OutputDevice(rst, initial_value=True) if rst is not None else None
        self.bl = PWMOutputDevice(bl, initial_value=1.0) if bl is not None else None

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed_hz
        self.spi.mode = 0

        self.rotation = rotation % 4
        self.invert = invert
        self.bgr = bgr
        self._init_panel()

    # ------------------------------------------------------------------ low level
    def _cmd(self, c, data=None):
        self.dc.off()
        self.spi.writebytes([c])
        if data:
            self.dc.on()
            self.spi.writebytes(list(data))

    def _reset(self):
        if self.rst is None:
            return
        self.rst.on()
        time.sleep(0.05)
        self.rst.off()
        time.sleep(0.05)
        self.rst.on()
        time.sleep(0.15)

    def _init_panel(self):
        self._reset()
        c = self._cmd
        c(0xE0, [0x00, 0x03, 0x09, 0x08, 0x16, 0x0A, 0x3F, 0x78, 0x4C, 0x09, 0x0A, 0x08, 0x16, 0x1A, 0x0F])
        c(0xE1, [0x00, 0x16, 0x19, 0x03, 0x0F, 0x05, 0x32, 0x45, 0x46, 0x04, 0x0E, 0x0D, 0x35, 0x37, 0x0F])
        c(0xC0, [0x17, 0x15])            # power control 1
        c(0xC1, [0x41])                  # power control 2
        c(0xC5, [0x00, 0x12, 0x80])      # VCOM
        c(0x3A, [0x66])                  # 18-bit colour (required for SPI)
        c(0xB0, [0x80])                  # interface mode
        c(0xB1, [0xA0])                  # frame rate (60 Hz)
        c(0xB4, [0x02])                  # display inversion control
        c(0xB6, [0x02, 0x02])            # display function control
        c(0xE9, [0x00])                  # set image function
        c(0xF7, [0xA9, 0x51, 0x2C, 0x82])  # adjust control 3
        c(0x21 if self.invert else 0x20)
        self.set_rotation(self.rotation)
        c(0x11)                          # sleep out
        time.sleep(0.12)
        c(0x29)                          # display on
        time.sleep(0.02)

    # ------------------------------------------------------------------ public API
    def set_rotation(self, r):
        self.rotation = r % 4
        m = _MADCTL[self.rotation]
        if not self.bgr:
            m ^= 0x08
        self._cmd(0x36, [m])
        self.width, self.height = (320, 480) if self.rotation in (0, 2) else (480, 320)

    def set_backlight(self, level):
        """0.0 - 1.0 (only if you wired LED to a GPIO and passed bl=...)."""
        if self.bl is not None:
            self.bl.value = max(0.0, min(1.0, level))

    def _window(self, x0, y0, x1, y1):
        self._cmd(0x2A, [x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])
        self._cmd(0x2B, [y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])
        self._cmd(0x2C)

    def display(self, img, x=0, y=0):
        """Draw a PIL image at (x, y). Full-screen image = whole display."""
        img = img.convert("RGB")
        w, h = img.size
        self._window(x, y, x + w - 1, y + h - 1)
        self.dc.on()
        self.spi.writebytes2(img.tobytes())

    def fill(self, rgb=(0, 0, 0)):
        self.display(Image.new("RGB", (self.width, self.height), rgb))

    def close(self):
        self.spi.close()


# ---------------------------------------------------------------------- demo
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


if __name__ == "__main__":
    lcd = ILI9488(rotation=1)          # 480x320 landscape
    W, H = lcd.width, lcd.height

    # 1) colour test: if RED and BLUE look swapped, use ILI9488(bgr=False)
    #    if colours look inverted (white<->black), use ILI9488(invert=True)
    img = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(img)
    bars = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    for i, col in enumerate(bars):
        d.rectangle([i * W // 4, 0, (i + 1) * W // 4, H // 2], fill=col)
    d.text((10, H // 2 + 10), "ILI9488 on Raspberry Pi 5", font=_font(28), fill="white")
    d.text((10, H // 2 + 50), "Bars: red, green, blue, white", font=_font(20), fill="yellow")
    lcd.display(img)
    time.sleep(4)

    # 2) live clock with partial updates (fast)
    lcd.fill((0, 0, 40))
    big = _font(72)
    try:
        while True:
            t = Image.new("RGB", (W, 100), (0, 0, 40))
            ImageDraw.Draw(t).text((20, 10), time.strftime("%H:%M:%S"), font=big, fill="white")
            lcd.display(t, 0, 110)
            time.sleep(0.2)
    except KeyboardInterrupt:
        lcd.fill((0, 0, 0))
        lcd.close()
