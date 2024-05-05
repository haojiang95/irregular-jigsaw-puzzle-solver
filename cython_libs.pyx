import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find_neighbors_within_radius_worker(cnp.ndarray[cnp.float64_t, ndim=2] points,
                                                    cnp.ndarray[cnp.float64_t, ndim=2] results,
                                                    const int center_index,
                                                    const double radius_squared,
                                                    int *num_results, const int start_index,
                                                    const int end_index,
                                                    const int delta):
    cdef double dx
    cdef double dy
    cdef int i = start_index
    while delta * i < delta * end_index:
        dx = points[i, 0] - points[center_index, 0]
        dy = points[i, 1] - points[center_index, 1]
        if dx * dx + dy * dy >= radius_squared:
            break
        results[num_results[0], 0] = points[i, 0]
        results[num_results[0], 1] = points[i, 1]
        num_results[0] += 1
        i += delta
    return i

@cython.boundscheck(False)
@cython.wraparound(False)
def find_neighbors_within_radius(cnp.ndarray[cnp.float64_t, ndim=2] points, const int center_index,
                                 const double radius):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] results = np.empty_like(points)
    cdef int num_results = 0
    cdef int num_points = len(points)
    cdef double radius_squared = radius ** 2
    cdef double dx
    cdef double dy

    cdef int i = find_neighbors_within_radius_worker(points, results, center_index, radius_squared, &num_results,
                                                     center_index, num_points, 1)

    if i == num_points:
        i = find_neighbors_within_radius_worker(points, results, center_index, radius_squared, &num_results,
                                                0, center_index, 1)

    i = find_neighbors_within_radius_worker(points, results, center_index, radius_squared, &num_results,
                                            center_index - 1, i if i <= center_index else -1, -1)

    if i == -1:
        find_neighbors_within_radius_worker(points, results, center_index, radius_squared, &num_results,
                                            num_points - 1, center_index, -1)

    return results[:num_results]
