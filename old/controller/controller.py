from pynput import keyboard
from threading import Event
from queue import Queue

from utils import console


class Controller:
    p = "[cyan1]keybrd[/cyan1]:"
    active_keys = []

    def __init__(self, kill_event: Event, key_queue: Queue) -> None:
        self.killed = kill_event
        self.queue = key_queue

    def run(self) -> None:
        with keyboard.Events() as events:
            while not self.killed.is_set():        
                event = events.get(1.0)
                if event:
                    if event.key not in self.active_keys:
                        self.active_keys.append(event.key)
                    else:
                        self.active_keys.remove(event.key)
                        continue

                    if event.key == keyboard.Key.left or event.key == keyboard.Key.right:
                        self.random(event.key)
                    else:
                        try:
                            match event.key.char: # type: ignore
                                case "l":
                                    self.loop()
                                case "b":
                                    self.rewind()
                                case "f" | "m" | "," | ".":
                                    self.volume(event.key)
                                case _:
                                    # console.log(f"{self.p} no match {event.key}")
                                    pass
                        except AttributeError:
                            pass
                            # console.log(
                            #     f"{self.p} special key pressed: {event.key}",
                            # )

    def volume(self, key):
        match key.char:
            case "f":
                self.queue.put("FADE")
            case "m":
                self.queue.put("MUTE")
            case ",":
                self.queue.put("VOL DOWN")
            case ".":
                self.queue.put("VOL UP")
            case _:
                console.log(f"{self.p} [bold orange]whoops, no match")
                pass

    def rewind(
        self,
    ):
        self.queue.put("BACK")

    def loop(
        self,
    ):
        self.queue.put("LOOP")

    def random(self, key):
        console.log(
            f"{self.p} [pale_turquoise1][RANDOM] [/pale_turquoise1] key pressed {key}"
        )
        self.queue.put("LOOP")
