import os

@property
def device_id(self) -> str:
    return os.environ.get("MACHINE_LEARNING_DEVICE_ID", "0")
