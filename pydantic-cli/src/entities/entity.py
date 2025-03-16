class Entity:
    def __init__(self, id: int, name: str):
        self.id: int = id
        self.name: str = name

    def __repr__(self) -> str:
        return f"Entity(id={self.id}, name='{self.name}')"

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}
