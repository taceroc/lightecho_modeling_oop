from numba import jit

class Shape:
    def __init__(self, a, b):
        self.a = a
        self.b = b  # Initialize your shape

    # @staticmethod
    @jit(nopython=True)
    def cal_area(self):
        print("area")
        return self.a*self.b

# Example usage
shape = Shape(2,2)
shape.cal_area()  # Call the cal_area method
