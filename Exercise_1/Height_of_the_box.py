import numpy as np

def find_height (data_cloud,floor_mask,normal,coeff):

    floor_point = np.zeros(3)
    for i in range(0, 1000):
        rand_floor_pointx = np.random.randint(0, floor_mask.shape[0], size=1)
        rand_floor_pointy = np.random.randint(0, floor_mask.shape[1], size=1)
        if floor_mask[rand_floor_pointx, rand_floor_pointy, 2] != 0:
            floor_point[0] = data_cloud[rand_floor_pointx, rand_floor_pointy, 0]
            floor_point[1] = data_cloud[rand_floor_pointx, rand_floor_pointy, 1]
            floor_point[2] = data_cloud[rand_floor_pointx, rand_floor_pointy, 2]
            break
        else:
            continue

#distance from a point to a plane - determinant of ax0+by0+cz0+d by sqrt of a^2+b^2+c^2 where a,b,c is a normal vector

    a = normal[0]
    b = normal[1]
    c = normal[2]
    num = abs((a * floor_point[0]) + (b * floor_point[1]) + (c * floor_point[2]) - coeff)
    den = np.sqrt((a * a) + (b * b) + (c * c))
    height = (num / den)
    print('Height by second method: ',height)
