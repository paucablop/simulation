from dataclasses import dataclass

@dataclass
class Agent:
    color: int
    index: int
    location: int
    satisfaction: float

    def move(self, threshold:float):
        if self.satisfaction >= threshold:
            return True
        else:
            return False