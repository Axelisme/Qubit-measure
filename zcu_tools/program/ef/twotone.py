from qick import AveragerProgram
from .base import BaseEFProgram


class EFProgram(AveragerProgram, BaseEFProgram):
    def initialize(self):
        return BaseEFProgram.initialize(self)
