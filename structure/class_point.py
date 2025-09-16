class Point:
    def __init__(self, vertices):
        self.x = vertices[0]
        self.y = vertices[1]
        self.z = vertices[2]

    def __repr__(self):
        return "x: {} y: {} z: {}".format(self.x, self.y, self.z)

