"""
Kernels to evaluate Green's functions.
"""
import numpy as np

try:
    from numba import njit, prange
except ImportError:
    print('Numba not found')

try:
    import PySTKFMM
except ImportError:
    pass


def oseen_tensor(r_vectors, eta=1.0, reg=5e-3, epsilon_distance=1e-5, input_format='r'):
    """
    Build the Oseen tensor for N points (sources and targets).
    Set to zero diagonal terms.

    G = f(r) * I + g(r) * (r.T*r)

    Input:
      r_vectors = coordinates.
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
      input_format = 'r' or 'xyz'.
        If 'r' r_vectors = (r_1, r_2, ..., r_N) with r_1 = (x_1, y_1, z_1).
        If 'xyz' r_vector = (x_all_points, y_all_points, z_all_points).

    Output:
      G = Oseen tensor with dimensions (3*num_points) x (3*num_points).

      Output with format
      if input_format == 'r':
            | G_11 G_12 ...|              | G(1,2)_xx G(1,2)_xy G(1,2)_xz|
        G = | G_21 G_22 ...|  with G_12 = | G(1,2)_yx G(1,2)_yy G(1,2)_yz|
            | ...          |              | G(1,2)_zx G(1,2)_zy G(1,2)_zz|
        and G_11 = 0

      if input_format == 'xyz':
            | G_xx G_xy G_xz|
        G = | G_yx G_yy G_yz|
            | G_zx G_zy G_zz|
    """
    N = r_vectors.size // 3
    if input_format == 'r':
        rx = r_vectors[:, 0]
        ry = r_vectors[:, 1]
        rz = r_vectors[:, 2]
    else:
        rx = r_vectors[0 * N:1 * N]
        ry = r_vectors[1 * N:2 * N]
        rz = r_vectors[2 * N:3 * N]

    # Compute vectors between points and distance
    drx = rx - rx[:, None]
    dry = ry - ry[:, None]
    drz = rz - rz[:, None]
    dr = np.sqrt(drx**2 + dry**2 + drz**2)

    # Compute scalar functions f(r) and g(r)
    sel = dr > epsilon_distance
    fr = np.zeros_like(dr)
    gr = np.zeros_like(dr)
    fr[sel] = 1.0 / (8.0 * np.pi * eta * dr[sel])
    gr[sel] = 1.0 / (8.0 * np.pi * eta * dr[sel]**3)

    # Add regularization term for points closer than epsilon_distance
    sel2 = np.invert(sel)
    fr[sel2] = 1.0 / (8.0 * np.pi * eta * np.sqrt(dr[sel2]**2 + reg**2))
    gr[sel2] = 1.0 / (8.0 * np.pi * eta * np.sqrt(dr[sel2]**2 + reg**2)**3)

    # Compute matrix of size 3N \times 3N
    G = np.zeros((r_vectors.size, r_vectors.size))
    if input_format == 'r':
        G[0::3, 0::3] = fr + gr * drx * drx
        G[0::3, 1::3] = gr * drx * dry
        G[0::3, 2::3] = gr * drx * drz

        G[1::3, 0::3] = gr * dry * drx
        G[1::3, 1::3] = fr + gr * dry * dry
        G[1::3, 2::3] = gr * dry * drz

        G[2::3, 0::3] = gr * drz * drx
        G[2::3, 1::3] = gr * drz * dry
        G[2::3, 2::3] = fr + gr * drz * drz
    else:
        G[0 * N:1 * N, 0 * N:1 * N] = fr + gr * drx * drx
        G[0 * N:1 * N, 1 * N:2 * N] = gr * drx * dry
        G[0 * N:1 * N, 2 * N:3 * N] = gr * drx * drz
        G[1 * N:2 * N, 0 * N:1 * N] = gr * dry * drx
        G[1 * N:2 * N, 1 * N:2 * N] = fr + gr * dry * dry
        G[1 * N:2 * N, 2 * N:3 * N] = gr * dry * drz
        G[2 * N:3 * N, 0 * N:1 * N] = gr * drz * drx
        G[2 * N:3 * N, 1 * N:2 * N] = gr * drz * dry
        G[2 * N:3 * N, 2 * N:3 * N] = fr + gr * drz * drz
    return G


def oseen_tensor_source_target(r_source, r_target, eta=1.0, reg=5e-3, epsilon_distance=1e-5, input_format='r'):
    """
    Build the Oseen tensor for Ns sources and Nt targets.
    Set to zero diagonal terms.

    G = f(r) * I + g(r) * (r.T*r)

    Input:
      r_vectors = coordinates.
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
      input_format = 'r' or 'xyz'.
        If 'r' r_vectors = (r_1, r_2, ..., r_N) with r_1 = (x_1, y_1, z_1).
        If 'xyz' r_vector = (x_all_points, y_all_points, z_all_points).

    Output:
      G = Oseen tensor with dimensions (3*num_points_targets) x (3*num_points_sources).

    Output with format
    if input_format == 'r':
          | G_11 G_12 ...|              | G(1,2)_xx G(1,2)_xy G(1,2)_xz|
      G = | G_21 G_22 ...|  with G_12 = | G(1,2)_yx G(1,2)_yy G(1,2)_yz|
          | ...          |              | G(1,2)_zx G(1,2)_zy G(1,2)_zz|
      and G_11 = 0

    if input_format == 'xyz':
          | G_xx G_xy G_xz|
      G = | G_yx G_yy G_yz|
          | G_zx G_zy G_zz|
    """
    Ns = r_source.size // 3
    Nt = r_target.size // 3
    if input_format == 'r':
        rxs = r_source[:, 0]
        rys = r_source[:, 1]
        rzs = r_source[:, 2]
        rxt = r_target[:, 0]
        ryt = r_target[:, 1]
        rzt = r_target[:, 2]
    else:
        rxs = r_source[0 * Ns:1 * Ns]
        rys = r_source[1 * Ns:2 * Ns]
        rzs = r_source[2 * Ns:3 * Ns]
        rxt = r_target[0 * Nt:1 * Nt]
        ryt = r_target[1 * Nt:2 * Nt]
        rzt = r_target[2 * Nt:3 * Nt]

    # Compute vectors between points and distance
    drx = rxs - rxt[:, None]
    dry = rys - ryt[:, None]
    drz = rzs - rzt[:, None]
    dr = np.sqrt(drx**2 + dry**2 + drz**2)

    # Compute scalar functions f(r) and g(r)
    sel = dr > epsilon_distance
    fr = np.zeros_like(dr)
    gr = np.zeros_like(dr)
    fr[sel] = 1.0 / (8.0 * np.pi * eta * dr[sel])
    gr[sel] = 1.0 / (8.0 * np.pi * eta * dr[sel]**3)

    # Add regularization term for points closer than epsilon_distance
    sel2 = np.invert(sel)
    fr[sel2] = 1.0 / (8.0 * np.pi * eta * np.sqrt(dr[sel2]**2 + reg**2))
    gr[sel2] = 1.0 / (8.0 * np.pi * eta * np.sqrt(dr[sel2]**2 + reg**2)**3)

    # Compute matrix of size 3N \times 3N
    G = np.zeros((r_target.size, r_source.size))
    if input_format == 'r':
        G[0::3, 0::3] = fr + gr * drx * drx
        G[0::3, 1::3] = gr * drx * dry
        G[0::3, 2::3] = gr * drx * drz

        G[1::3, 0::3] = gr * dry * drx
        G[1::3, 1::3] = fr + gr * dry * dry
        G[1::3, 2::3] = gr * dry * drz

        G[2::3, 0::3] = gr * drz * drx
        G[2::3, 1::3] = gr * drz * dry
        G[2::3, 2::3] = fr + gr * drz * drz
    else:
        G[0 * Nt:1 * Nt, 0 * Ns:1 * Ns] = fr + gr * drx * drx
        G[0 * Nt:1 * Nt, 1 * Ns:2 * Ns] = gr * drx * dry
        G[0 * Nt:1 * Nt, 2 * Ns:3 * Ns] = gr * drx * drz
        G[1 * Nt:2 * Nt, 0 * Ns:1 * Ns] = gr * dry * drx
        G[1 * Nt:2 * Nt, 1 * Ns:2 * Ns] = fr + gr * dry * dry
        G[1 * Nt:2 * Nt, 2 * Ns:3 * Ns] = gr * dry * drz
        G[2 * Nt:3 * Nt, 0 * Ns:1 * Ns] = gr * drz * drx
        G[2 * Nt:3 * Nt, 1 * Ns:2 * Ns] = gr * drz * dry
        G[2 * Nt:3 * Nt, 2 * Ns:3 * Ns] = fr + gr * drz * drz
    return G


@njit(parallel=True, fastmath=True)
def oseen_kernel_times_density_numba(r_vectors, density, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """
    Oseen tensor product with a density force for N points (sources and targets).
    Set to zero diagonal terms.

    u_i = sum_j G_ij * density_j

    Input:
      r_vectors = coordinates.
      normal = vector used to contract the Stresslet (in general
            this will be the normal vector of a surface).
      density = vector used to contract the Stresslet (in general
            this will be a double layer potential).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
      u = velocity, dimension (3*num_points)
    """
    # Variables
    N = r_vectors.size // 3
    factor = 1.0 / (8.0 * np.pi * eta)
    r_vectors = r_vectors.reshape(N, 3)
    density = density.reshape(N, 3)
    u = np.zeros((N, 3))

    # Loop over targets
    for xn in prange(N):
        for yn in range(xn):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)
                # continue
            fr = factor / r_norm
            gr = factor / r_norm**3
            Mxx = fr + gr * x * x
            Mxy = gr * x * y
            Mxz = gr * x * z
            Myy = fr + gr * y * y
            Myz = gr * y * z
            Mzz = fr + gr * z * z

            u[xn, 0] += Mxx * density[yn, 0] + Mxy * density[yn, 1] + Mxz * density[yn, 2]
            u[xn, 1] += Mxy * density[yn, 0] + Myy * density[yn, 1] + Myz * density[yn, 2]
            u[xn, 2] += Mxz * density[yn, 0] + Myz * density[yn, 1] + Mzz * density[yn, 2]

        for yn in range(xn + 1, N):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            fr = factor / r_norm
            gr = factor / r_norm**3
            Mxx = fr + gr * x * x
            Mxy = gr * x * y
            Mxz = gr * x * z
            Myy = fr + gr * y * y
            Myz = gr * y * z
            Mzz = fr + gr * z * z

            u[xn, 0] += Mxx * density[yn, 0] + Mxy * density[yn, 1] + Mxz * density[yn, 2]
            u[xn, 1] += Mxy * density[yn, 0] + Myy * density[yn, 1] + Myz * density[yn, 2]
            u[xn, 2] += Mxz * density[yn, 0] + Myz * density[yn, 1] + Mzz * density[yn, 2]

    return u.flatten()


@njit(parallel=True, fastmath=True)
def oseen_kernel_source_target_numba(r_source, r_target, density, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """
    Oseen tensor product with a density force for N points (sources and targets).
    Set to zero diagonal terms.

    u_i = sum_j G_ij * density_j

    Input:
      r_vectors = coordinates.
      density = vector used to contract the Stresslet (in general this will be a double layer potential).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
      u = velocity, dimension (3*num_points)
    """
    # Variables
    Nsource = r_source.size // 3
    Ntarget = r_target.size // 3
    factor = 1.0 / (8.0 * np.pi * eta)
    r_source = r_source.reshape(Nsource, 3)
    r_target = r_target.reshape(Ntarget, 3)
    density = density.reshape(Nsource, 3)
    u = np.zeros((Ntarget, 3))

    # Loop over targets
    for xn in prange(Ntarget):
        for yn in range(Nsource):
            x = r_target[xn, 0] - r_source[yn, 0]
            y = r_target[xn, 1] - r_source[yn, 1]
            z = r_target[xn, 2] - r_source[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            fr = factor / r_norm
            gr = factor / r_norm**3
            Mxx = fr + gr * x * x
            Mxy = gr * x * y
            Mxz = gr * x * z
            Myy = fr + gr * y * y
            Myz = gr * y * z
            Mzz = fr + gr * z * z

            u[xn, 0] += Mxx * density[yn, 0] + Mxy * density[yn, 1] + Mxz * density[yn, 2]
            u[xn, 1] += Mxy * density[yn, 0] + Myy * density[yn, 1] + Myz * density[yn, 2]
            u[xn, 2] += Mxz * density[yn, 0] + Myz * density[yn, 1] + Mzz * density[yn, 2]

    return u.flatten()


def oseen_kernel_source_target_pycuda(r_source, r_target, density, eta=1.0, epsilon_distance=1e-5):
    """

    """
    return kernels_pycuda.oseen_kernel_source_target_pycuda(r_source,
                                                            r_target,
                                                            density,
                                                            eta=eta,
                                                            epsilon_distance=epsilon_distance)


@njit(parallel=True, fastmath=True)
def rotlet_kernel_source_target_numba(r_source, r_target, density, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """
    Oseen tensor product with a density force for N points (sources and targets).
    Set to zero diagonal terms.

    u_i = sum_j G_ij * density_j

    Input:
      r_vectors = coordinates.
      normal = vector used to contract the Stresslet (typically the normal vector of a surface).
      density = vector used to contract the Stresslet (typically a double layer potential).
      eta = (default 1.0) viscosity
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
      u = velocity, dimension (3*num_points)
    """
    # Variables
    Nsource = r_source.size // 3
    Ntarget = r_target.size // 3
    factor = 1.0 / (8.0 * np.pi * eta)
    r_source = r_source.reshape(Nsource, 3)
    r_target = r_target.reshape(Ntarget, 3)
    density = density.reshape(Nsource, 3)
    u = np.zeros((Ntarget, 3))

    # Loop over targets
    for xn in prange(Ntarget):
        for yn in range(Nsource):
            x = r_target[xn, 0] - r_source[yn, 0]
            y = r_target[xn, 1] - r_source[yn, 1]
            z = r_target[xn, 2] - r_source[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            fr = factor / r_norm**3
            Mxy = fr * z
            Mxz = -fr * y
            Myx = -fr * z
            Myz = fr * x
            Mzx = fr * y
            Mzy = -fr * x

            u[xn, 0] += Mxy * density[yn, 1] + Mxz * density[yn, 2]
            u[xn, 1] += Myx * density[yn, 0] + Myz * density[yn, 2]
            u[xn, 2] += Mzx * density[yn, 0] + Mzy * density[yn, 1]

    return u.flatten()


def stresslet_kernel(r_vectors, eta=1.0, reg=5e-3, epsilon_distance=1e-5, input_format='r'):
    """
    Build the Stresslet tensor for N points (sources and targets).
    Set to zero diagonal terms.

    S_ijk = -(3/(4*pi)) * r_i * r_j * r_k / r**5

    Input:
      r_vectors = coordinates.
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
      input_format = 'r' or 'xyz'. Only 'r' is implemented.
                     If 'r' r_vectors = (r_1, r_2, ..., r_N) with r_1 = (x_1, y_1, z_1).
                     If 'xyz' r_vector = (x_all_points, y_all_points, z_all_points).

    Output:
      S = Stresslet tensor with dimensions (3*num_points) x (3*num_points) x (3)
          Output with format

           | S_11 S_12 ...|
       S = | S_12 S_12 ...|
           | ...          |

      with S_12 the stresslet between points (3x3x3) tensor between points 1 and 2.
    """

    N = r_vectors.size // 3
    if input_format == 'r':
        rx = r_vectors[:, 0]
        ry = r_vectors[:, 1]
        rz = r_vectors[:, 2]
    else:
        rx = r_vectors[0 * N:1 * N]
        ry = r_vectors[1 * N:2 * N]
        rz = r_vectors[2 * N:3 * N]

    # Compute vectors between points and distance
    drx = rx - rx[:, None]
    dry = ry - ry[:, None]
    drz = rz - rz[:, None]
    dr = np.sqrt(drx**2 + dry**2 + drz**2)

    # Compute scalar functions f(r) and g(r)
    sel = dr > epsilon_distance
    fr = np.zeros_like(dr)
    fr[sel] = -3.0 / (4.0 * np.pi * eta * dr[sel]**5)
    sel = dr <= epsilon_distance
    fr[sel] = -3.0 / (4.0 * np.pi * eta * np.power(dr[sel]**2 + reg**2, 2.5))

    # Compute stresslet
    S = np.zeros((r_vectors.size, r_vectors.size, 3))
    S[0::3, 0::3, 0] = fr * drx * drx * drx
    S[0::3, 0::3, 1] = fr * drx * drx * dry
    S[0::3, 0::3, 2] = fr * drx * drx * drz
    S[0::3, 1::3, 0] = fr * drx * dry * drx
    S[0::3, 1::3, 1] = fr * drx * dry * dry
    S[0::3, 1::3, 2] = fr * drx * dry * drz
    S[0::3, 2::3, 0] = fr * drx * drz * drx
    S[0::3, 2::3, 1] = fr * drx * drz * dry
    S[0::3, 2::3, 2] = fr * drx * drz * drz
    S[1::3, 0::3, 0] = fr * dry * drx * drx
    S[1::3, 0::3, 1] = fr * dry * drx * dry
    S[1::3, 0::3, 2] = fr * dry * drx * drz
    S[1::3, 1::3, 0] = fr * dry * dry * drx
    S[1::3, 1::3, 1] = fr * dry * dry * dry
    S[1::3, 1::3, 2] = fr * dry * dry * drz
    S[1::3, 2::3, 0] = fr * dry * drz * drx
    S[1::3, 2::3, 1] = fr * dry * drz * dry
    S[1::3, 2::3, 2] = fr * dry * drz * drz
    S[2::3, 0::3, 0] = fr * drz * drx * drx
    S[2::3, 0::3, 1] = fr * drz * drx * dry
    S[2::3, 0::3, 2] = fr * drz * drx * drz
    S[2::3, 1::3, 0] = fr * drz * dry * drx
    S[2::3, 1::3, 1] = fr * drz * dry * dry
    S[2::3, 1::3, 2] = fr * drz * dry * drz
    S[2::3, 2::3, 0] = fr * drz * drz * drx
    S[2::3, 2::3, 1] = fr * drz * drz * dry
    S[2::3, 2::3, 2] = fr * drz * drz * drz

    return S


def stresslet_kernel_times_normal(r_vectors, normal, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """
    Build the Stresslet tensor contracted with a vector for N points (sources and targets).
    Set to zero diagonal terms.

    S_ij = sum_k -(3/(4*pi)) * r_i * r_j * r_k * normal_k / r**5

    Input:
      r_vectors = coordinates.
      normal = vector used to contract the Stresslet (in general this will be the normal vector of a surface).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
      S_normal = Stresslet tensor contracted with a vector with dimensions (3*num_points) x (3*num_points).

    Output with format

                 | S_normal11 S_normal12 ...|
      S_normal = | S_normal21 S_normal22 ...|
                 | ...                      |

      with S_normal12 the stresslet between points r_1 and r_2.

      S_normal12 has dimensions 3 x 3.
    """
    rx = r_vectors[:, 0]
    ry = r_vectors[:, 1]
    rz = r_vectors[:, 2]

    # Compute vectors between points and distance
    drx = rx - rx[:, None]
    dry = ry - ry[:, None]
    drz = rz - rz[:, None]
    dr = np.sqrt(drx**2 + dry**2 + drz**2)

    # Compute scalar functions f(r) and g(r)
    sel = dr > epsilon_distance
    fr = np.zeros_like(dr)
    fr[sel] = 3.0 / (4.0 * np.pi * eta * dr[sel]**5)
    sel = dr <= epsilon_distance
    fr[sel] = 3.0 / (4.0 * np.pi * eta * np.power(dr[sel]**2 + reg**2, 2.5))

    # Contract r_k with vector
    normal = np.reshape(normal, (normal.size // 3, 3))
    contraction = drx * normal[:, 0] + dry * normal[:, 1] + drz * normal[:, 2]

    # Compute contracted stresslet
    Snormal = np.zeros((r_vectors.size, r_vectors.size))
    Snormal[0::3, 0::3] = fr * drx * drx * contraction
    Snormal[0::3, 1::3] = fr * drx * dry * contraction
    Snormal[0::3, 2::3] = fr * drx * drz * contraction

    Snormal[1::3, 0::3] = fr * dry * drx * contraction
    Snormal[1::3, 1::3] = fr * dry * dry * contraction
    Snormal[1::3, 2::3] = fr * dry * drz * contraction

    Snormal[2::3, 0::3] = fr * drz * drx * contraction
    Snormal[2::3, 1::3] = fr * drz * dry * contraction
    Snormal[2::3, 2::3] = fr * drz * drz * contraction

    return Snormal


@njit(parallel=False, fastmath=True)
def stresslet_kernel_times_normal_numba(r_vectors, normal, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """
    Build the Stresslet tensor contracted with a vector for N points (sources and targets).
    Set to zero diagonal terms.

    S_ij = sum_k -(3/(4*pi)) * r_i * r_j * r_k

    Input:
      r_vectors = coordinates.
      normal = vector used to contract the Stresslet (in general this will be the normal vector of a surface).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
    S_normal = Stresslet tensor contracted with a vector with dimensions (3*num_points) x (3*num_points).

    Output with format

                | S_normal11 S_normal12 ...|
     S_normal = | S_normal21 S_normal22 ...|
                | ...                      |

    with S_normal12 the stresslet between points r_1 and r_2.

    S_normal12 has dimensions 3 x 3.
    """
    # Variables
    N = r_vectors.size // 3
    factor = -3.0 / (4.0 * np.pi * eta)
    r_vectors = r_vectors.reshape(N, 3)
    normal = normal.reshape(N, 3)
    Snormal = np.zeros((3 * N, 3 * N))

    # Loop over targets
    for xn in prange(N):
        for yn in range(xn):
            r = r_vectors[xn] - r_vectors[yn]
            r_norm = np.linalg.norm(r)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            S = (factor * np.dot(r, normal[yn]) / r_norm**5) * np.outer(r, r)
            Snormal[3 * xn:3 * (xn + 1), 3 * yn:3 * (yn + 1)] = S

        for yn in range(xn + 1, N):
            r = r_vectors[xn] - r_vectors[yn]
            r_norm = np.linalg.norm(r)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            S = (factor * np.dot(r, normal[yn]) / r_norm**5) * np.outer(r, r)
            Snormal[3 * xn:3 * (xn + 1), 3 * yn:3 * (yn + 1)] = S

    return Snormal


@njit(parallel=True, fastmath=True)
def stresslet_kernel_times_normal_times_density_numba(r_vectors,
                                                      normal,
                                                      density,
                                                      eta=1.0,
                                                      reg=5e-3,
                                                      epsilon_distance=1e-5):
    """
    Build the Stresslet tensor contracted with two vectors for N points (sources and targets).
    Set to zero diagonal terms.

    S_i = sum_jk -(3/(4*pi)) * r_i * r_j * r_k * density_j * normal_k / r**5

    Input:
      r_vectors = coordinates.
      normal = vector used to contract the Stresslet (in general this will be the normal vector of a surface).
      density = vector used to contract the Stresslet (in general this will be a double layer potential).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.

    Output:
      S_normal = Stresslet tensor contracted with two vectors with dimensions (3*num_points).
    """
    # Variables
    N = r_vectors.size // 3
    factor = -3.0 / (4.0 * np.pi * eta)
    r_vectors = r_vectors.reshape(N, 3)
    normal = normal.reshape(N, 3)
    density = density.reshape(N, 3)
    Sdn = np.zeros((N, 3))

    # Loop over targets
    for xn in prange(N):
        for yn in range(xn):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            r_inv5 = 1.0 / r_norm**5
            f0 = factor * (x * density[yn, 0] + y * density[yn, 1] +
                           z * density[yn, 2]) * (x * normal[yn, 0] + y * normal[yn, 1] + z * normal[yn, 2]) * r_inv5
            Sdn[xn, 0] += f0 * x
            Sdn[xn, 1] += f0 * y
            Sdn[xn, 2] += f0 * z

        for yn in range(xn + 1, N):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            r_inv5 = 1.0 / r_norm**5
            f0 = factor * (x * density[yn, 0] + y * density[yn, 1] +
                           z * density[yn, 2]) * (x * normal[yn, 0] + y * normal[yn, 1] + z * normal[yn, 2]) * r_inv5
            Sdn[xn, 0] += f0 * x
            Sdn[xn, 1] += f0 * y
            Sdn[xn, 2] += f0 * z

    return Sdn.flatten()


@njit(parallel=True, fastmath=True)
def stresslet_kernel_source_target_numba(r_source, r_target, normal, density, eta=1.0, reg=5e-3, epsilon_distance=1e-5):
    """

    """
    # Variables
    Nsource = r_source.size // 3
    Ntarget = r_target.size // 3
    factor = -3.0 / (4.0 * np.pi * eta)
    r_source = r_source.reshape((Nsource, 3))
    r_target = r_target.reshape(Ntarget, 3)
    normal = normal.reshape(Nsource, 3)
    density = density.reshape(Nsource, 3)
    Sdn = np.zeros((Ntarget, 3))

    # Loop over targets
    for xn in prange(Ntarget):
        for yn in range(Nsource):
            x = r_target[xn, 0] - r_source[yn, 0]
            y = r_target[xn, 1] - r_source[yn, 1]
            z = r_target[xn, 2] - r_source[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            r_inv5 = 1.0 / r_norm**5
            f0 = factor * (x * density[yn, 0] + y * density[yn, 1] +
                           z * density[yn, 2]) * (x * normal[yn, 0] + y * normal[yn, 1] + z * normal[yn, 2]) * r_inv5
            Sdn[xn, 0] += f0 * x
            Sdn[xn, 1] += f0 * y
            Sdn[xn, 2] += f0 * z

    return Sdn.flatten()


@njit(parallel=True, fastmath=True)
def traction_kernel_times_normal_times_density_numba(r_vectors,
                                                     normal,
                                                     density,
                                                     eta=1.0,
                                                     reg=5e-3,
                                                     epsilon_distance=1e-5):
    """

    """
    # Variables
    N = r_vectors.size // 3
    factor = -3.0 / (4.0 * np.pi * eta)
    r_vectors = r_vectors.reshape(N, 3)
    normal = normal.reshape(N, 3)
    density = density.reshape(N, 3)
    Sdn = np.zeros((N, 3))
    eta = 1.0

    # Loop over targets
    for xn in prange(N):
        for yn in range(xn):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            r_inv5 = 1.0 / r_norm**5
            f0 = factor * (x * density[yn, 0] + y * density[yn, 1] +
                           z * density[yn, 2]) * (x * normal[xn, 0] + y * normal[xn, 1] + z * normal[xn, 2]) * r_inv5
            Sdn[xn, 0] += f0 * x
            Sdn[xn, 1] += f0 * y
            Sdn[xn, 2] += f0 * z

        for yn in range(xn + 1, N):
            x = r_vectors[xn, 0] - r_vectors[yn, 0]
            y = r_vectors[xn, 1] - r_vectors[yn, 1]
            z = r_vectors[xn, 2] - r_vectors[yn, 2]
            r_norm = np.sqrt(x**2 + y**2 + z**2)
            if r_norm < epsilon_distance:
                r_norm = np.sqrt(r_norm**2 + reg**2)

            r_inv5 = 1.0 / r_norm**5
            f0 = factor * (x * density[yn, 0] + y * density[yn, 1] +
                           z * density[yn, 2]) * (x * normal[xn, 0] + y * normal[xn, 1] + z * normal[xn, 2]) * r_inv5
            Sdn[xn, 0] += f0 * x
            Sdn[xn, 1] += f0 * y
            Sdn[xn, 2] += f0 * z

    return Sdn.flatten()


@njit(parallel=True, fastmath=True)
def complementary_kernel_times_density_numba(r_vectors, normal, density):
    """

    """
    # Variables
    N = normal.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    density = density.reshape((N, 3))
    u = np.zeros((N, 3))

    # Loop over targets
    for xn in prange(N):
        for yn in range(N):
            u[xn] += normal[xn] * (normal[yn, 0] * density[yn, 0] + normal[yn, 1] * density[yn, 1] +
                                   normal[yn, 2] * density[yn, 2])

    return u.flatten()


def complementary_kernel(r_vectors, normal):
    """

    """
    # Variables
    N = normal.size // 3
    r_vectors = r_vectors.reshape((N, 3))
    Nk = np.outer(normal.flatten(), normal.flatten())

    return Nk


def get_containing_box(coords):
    r_min = np.min(coords[0])
    r_max = np.max(coords[0])
    for i in range(1, len(coords)):
        r_min = np.min((np.min(coords[i]), r_min))
        r_max = np.max((np.max(coords[i]), r_max))

    L = 2 * 1.05 * np.max((np.fabs(r_min), np.fabs(r_max)))

    return np.array([-L / 2, -L / 2, -L / 2]), L


def oseen_kernel_source_target_stkfmm(r_source,
                                      r_target,
                                      density,
                                      eta=1.0,
                                      reg=5e-3,
                                      epsilon_distance=1e-5,
                                      *args,
                                      **kwargs):
    """
    Oseen tensor product with a density force for N points (sources and targets).
    Set to zero diagonal terms.

    u_i = sum_j G_ij * density_j

    Input:
      r_vectors = coordinates.
      density = vector used to contract the Stresslet (in general
            this will be a double layer potential).
      eta = (default 1.0) viscosity
      reg = (default 5e-3) regularization term
      epsilon_distance = (default 1e-10) set elements to zero for
                         distances < epsilon_distance.

    Output:
      u = velocity, dimension (3*num_points)

    This function uses the stkfmm library.
    """
    # Get fmm
    fmm_PVel = kwargs.get('fmm_PVel')
    nsrc = r_source.size // 3
    ntrg = r_target.size // 3
    r_source = r_source.reshape(nsrc, 3)
    r_target = r_target.reshape(ntrg, 3)
    density = density.reshape(density.size // 3, 3)

    # Set tree if necessary
    build_tree = True
    if np.array_equal(fmm_PVel.r_source_old, r_source) and \
       np.array_equal(fmm_PVel.r_target_old, r_target):
        build_tree = False
    if build_tree:
        # Build tree in STKFMM
        origin, L = get_containing_box((r_source, r_target))
        fmm_PVel.set_box(origin, L)
        fmm_PVel.set_points(r_source, r_target, np.zeros(0))
        fmm_PVel.setup_tree(PySTKFMM.KERNEL.Stokes)

        # Save vectors for next call
        fmm_PVel.r_source_old = np.copy(r_source)
        fmm_PVel.r_target_old = np.copy(r_target)

    # Evaluate fmm
    trg_value = np.zeros((ntrg, 3))
    fmm_PVel.clear_fmm(PySTKFMM.KERNEL.Stokes)
    fmm_PVel.evaluate_fmm(PySTKFMM.KERNEL.Stokes, density, trg_value, np.zeros(0))

    # Compute velocity
    return (trg_value / eta).flatten()


def stresslet_kernel_source_target_stkfmm(r_source,
                                          r_target,
                                          normal,
                                          density,
                                          eta=1.0,
                                          reg=5e-3,
                                          epsilon_distance=1e-5,
                                          *args,
                                          **kwargs):
    """

    """
    # Get fmm
    fmm_PVel = kwargs.get('fmm_PVel')
    nsrc = r_source.size // 3
    ntrg = r_target.size // 3

    r_source = r_source.reshape(nsrc, 3)
    r_target = r_target.reshape(ntrg, 3)
    normal = normal.reshape(nsrc, 3)
    density = density.reshape(nsrc, 3)

    # Set tree if necessary
    build_tree = True
    if np.array_equal(fmm_PVel.r_source_old, r_source) and \
       np.array_equal(fmm_PVel.r_target_old, r_target):
        build_tree = False
    if build_tree:
        # Build tree in STKFMM
        origin, L = get_containing_box((r_source, r_target))
        fmm_PVel.set_box(origin, L)
        fmm_PVel.set_points(np.zeros(0), r_target, r_source)
        # FIXME: Get DL Stokes kernel that doesn't calculate pressure
        fmm_PVel.setup_tree(PySTKFMM.KERNEL.PVel)

        # Save vectors for next call
        fmm_PVel.r_source_old = np.copy(r_source)
        fmm_PVel.r_target_old = np.copy(r_target)

    # Set density with right format
    # MIND FACTOR TWO IN THE DOUBLE LAYER DENSITY!!!
    trg_value = np.zeros((ntrg, 4))
    src_DL_value = np.einsum('ij,ik->ijk', normal, density).reshape((nsrc, 9)) * 2.0

    # Evaluate fmm; format pressure = trg_value[:,0], velocity = trg_value[:,1:4]
    fmm_PVel.clear_fmm(PySTKFMM.KERNEL.PVel)
    fmm_PVel.evaluate_fmm(PySTKFMM.KERNEL.PVel, np.zeros(0), trg_value, src_DL_value)

    # Compute velocity
    vel = trg_value[:, 1:4] / eta
    return vel.flatten()
