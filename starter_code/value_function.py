import numpy as np

class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = ex_space
        self.ey_space = ey_space
        self.etheta_space = etheta_space

    def copy_from(self, other):
        """
        Update the underlying value function storage with another value function
        """
        raise NotImplementedError

    def update(self, t, ex, ey, etheta, target_value):
        """
        Update the value function at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
            target_value: target value
        """
        raise NotImplementedError

    def __call__(self, t, ex, ey, etheta):
        """
        Get the value function results at given states
        Args:
            t: time step
            ex: x position error
            ey: y position error
            etheta: theta error
        Returns:
            value function results
        """
        raise NotImplementedError

    def copy(self):
        """
        Create a copy of the value function
        Returns:
            a copy of the value function
        """
        raise NotImplementedError

class GridValueFunction(ValueFunction):
    """
    Grid-based value function
    """
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
        # self.value_function = np.zeros((T, len(ex_space), len(ey_space), len(etheta_space)), dtype=np.float32)
        self.value_function = np.zeros((T, len(ex_space), len(ey_space), len(etheta_space)), dtype=np.float32)


    def copy_from(self, other):
        self.value_function = np.copy(other.value_function)

    def update(self, t, ex, ey, etheta, target_value):
        i = np.digitize(ex, self.ex_space) - 1
        j = np.digitize(ey, self.ey_space) - 1
        k = np.digitize(etheta, self.etheta_space) - 1
        self.value_function[t, i, j, k] = target_value

    def __call__(self, t, ex, ey, etheta):
        i = np.digitize(ex, self.ex_space) - 1
        j = np.digitize(ey, self.ey_space) - 1
        k = np.digitize(etheta, self.etheta_space) - 1
        return self.value_function[t, i, j, k]

    def copy(self):
        new_vf = GridValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new_vf.copy_from(self)
        return new_vf

# class FeatureValueFunction(ValueFunction):
#     """
#     Feature-based value function
#     """
#     # TODO: your implementation
#     raise NotImplementedError





