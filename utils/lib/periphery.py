"""
Small class to handle the periphery.
"""
import numpy as np
import lib.kernels as kernels


class Periphery(object):
    """
    Small class to handle a single body.
    """
    def __init__(self, location, orientation, reference_configuration, reference_normals, quadrature_weights):
        """
        Constructor. Take arguments like ...
        """
        # Location as np.array.shape = 3
        self.location = location
        # Orientation as Quaternion
        self.orientation = orientation
        # Number of blobs
        self.Nblobs = reference_configuration.size // 3
        # Reference configuration. Coordinates of blobs for quaternion [1, 0, 0, 0]
        # and location = np.array[0, 0, 0]) as a np.array.shape = (Nblobs, 3)
        # or np.array.shape = (Nblobs * 3)
        self.reference_configuration = np.reshape(reference_configuration, (self.Nblobs, 3))
        self.reference_normals = np.reshape(reference_normals, (self.Nblobs, 3))
        self.quadrature_weights = quadrature_weights.flatten()
        self.Nblobs = self.quadrature_weights.size
        # Name of body and type of body. A string or number
        self.name = None
        self.type = None
        self.rotation_matrix = None
        # Some default functions
        self.function_slip = np.zeros((self.Nblobs, 3))
        self.ID = None
        # Vectors for singularity subtractions
        self.ex = None
        self.ey = None
        self.ez = None
        self.density = np.zeros(3 * self.Nblobs)
        self.density_new = np.zeros(3 * self.Nblobs)

    def get_r_vectors(self, location=None, orientation=None):
        """
        Return the coordinates of the blobs.
        """
        # Get location and orientation
        if location is None:
            location = self.location
        if orientation is None:
            orientation = self.orientation

        # Compute blobs coordinates
        rotation_matrix = orientation.rotation_matrix()
        r_vectors = np.array([np.dot(rotation_matrix, vec) for vec in self.reference_configuration])
        r_vectors += location
        return r_vectors

    def get_normals(self, orientation=None):
        """
        Return the normals of the periphery.
        """
        # Get orientation
        if orientation is None:
            orientation = self.orientation

        # Compute blobs coordinates
        rotation_matrix = orientation.rotation_matrix()
        normals = np.array([np.dot(rotation_matrix, vec) for vec in self.reference_normals])
        return normals

    def get_singularity_subtraction_vectors(self, eta=1):

        # Compute correction for singularity subtractions
        r_vectors_blobs = self.get_r_vectors()
        normals = self.get_normals()
        quadrature_weights = self.quadrature_weights
        Nperiphery = quadrature_weights.size
        e = np.zeros((Nperiphery, 3))
        e[:, 0] = 1.0
        e *= quadrature_weights[:, None]
        self.ex = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, eta)
        e[:, :] = 0.0
        e[:, 1] = 1.0
        e *= quadrature_weights[:, None]
        self.ey = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, eta)
        e[:, :] = 0.0
        e[:, 2] = 1.0
        e *= quadrature_weights[:, None]
        self.ez = kernels.stresslet_kernel_times_normal_times_density_numba(r_vectors_blobs, normals, e, eta)

        return
