from abc import ABC, abstractmethod

class StructuralEquationModel(ABC):
    @staticmethod
    @abstractmethod
    def static(self):
        pass

    @staticmethod
    @abstractmethod
    def dynamic(self):
        pass