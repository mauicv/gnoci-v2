import numpy as np
import time
import gpiod
from gpiod.line import Direction, Bias


class ContactHWI:
    FEET_CONTACT_GPIO_CONFIG = {
        (6, 13): gpiod.LineSettings(
            direction=Direction.INPUT,
            bias=Bias.PULL_UP,
            active_low=True
        )
    }
    def __init__(self, gpio_chip="/dev/gpiochip0", consumer="feet"):
        self.gpio_lines = gpiod.request_lines(
            gpio_chip, consumer=consumer,
            config=self.FEET_CONTACT_GPIO_CONFIG
        )

    def get_contacts(self):
        values = self.gpio_lines.get_values([6, 13])
        return bool(values[0]), bool(values[1])
