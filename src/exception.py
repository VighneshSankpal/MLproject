import sys
from typing import Any
from logger import logging

class CustomException(Exception):
    def __init__(self, error_message: Any):
        _, _, exc_tb = sys.exc_info()

        if exc_tb:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.lineno = -1

        super().__init__(str(error_message))

    def __str__(self):
        return (
            f"Error occurred in file [{self.file_name}] "
            f"at line [{self.lineno}]: {super().__str__()}"
        )


try:
    1 / 0
except Exception as e:
    logging.info(e)
    raise CustomException(e)
