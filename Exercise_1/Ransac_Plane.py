import numpy as np

rands = np.ones((3, 3))

def  ransac(input_data, no_itertions, threshold,type):
    val = input_data.shape
    best_no_inliers = 0
    best_coeffd = 0
    best_normal = 0
    for i in range(no_itertions):
        coeff, normal = estimate(input_data)
        idx, dist_values = distance_mask(coeff,normal,input_data)
        no_inliers_ = no_of_inliers(dist_values,threshold)
        if no_inliers_ > best_no_inliers:
            best_coeffd = coeff
            best_no_inliers = no_inliers_
            best_normal = normal
            if type=='floor_2':
                best_mask= create_floor_2_mask(dist_values,threshold,idx,val)
            else:
                best_mask = create_mask(dist_values,threshold,idx,val)

        else :
            continue
    return best_no_inliers, best_coeffd, best_mask,best_normal

'''
def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]'''


def estimate (data):
    size = data.shape #(3,217088)
    rand = np.random.randint(0, size[1], 3) #3nos within 217088 [x,x,x]

    for i in range(3):
        init_points = data[:, rand[i]]
        if init_points[2]== 0: # to check Z values
            rand[i] = np.random.randint(0, size[1], 1)
        else:
            rands[:, i] = init_points
    normal = np.cross((rands[:, 0] - rands[:, 2]), (rands[:, 1] - rands[:, 2]))/ np.linalg.norm(np.cross((rands[:, 0] - rands[:, 2]), (rands[:, 1] - rands[:, 2])))
    coeff = np.dot(normal, rands[:, 2])
    return coeff, normal

def distance_mask(coeff, normal,data):
    data = data.T
   # print(data.shape[0])
    idx = np.zeros(data.shape[0],dtype=bool)
   # print(idx.shape)
   # for i in range (0, data.shape[0]):
    #    if data[i, 2] !=0:
     #       idx[i] = True

    idx = data[:,2]!=0
    d_mask = np.dot(data[idx], normal)- coeff
    return idx, d_mask

def no_of_inliers(d_mask, threshold):
   #print(d_mask.shape)
   count = 0
   #for i in range(0, d_mask.shape[0]):
    #   if np.abs(d_mask[i]) <=threshold:
   #        count += 1
   val= np.abs(d_mask)<= threshold
   count = np.count_nonzero(val)
   #print(count)
   return count

def create_mask(dist_values,threshold,idx,val):
    mask = np.zeros((3, val[1]))
    #print(dist_values.shape)
    init_mask = np.zeros_like(dist_values,dtype=bool)
    #print(init_mask.shape)
  #  for i in range(0, dist_values.shape[0]):
   #     if np.abs(dist_values[i])<= threshold:
    #        init_mask[i] = True
    init_mask[:] = np.abs(dist_values)<= threshold

    for i in range(3):
        mask[i, idx] = init_mask
    return mask


def create_floor_2_mask(dist_values,threshold,idx,val):
    mask = np.zeros((3, val[1]))

    for i in range(3):
        mask[i, idx] = dist_values
    return mask


'''
def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold'''

