import numpy as np

def quarternion_mult(q, r):
    return [r[0]*q[0]-r[1]*q[1]-r[2]*q[2]-r[3]*q[3],
            r[0]*q[1]+r[1]*q[0]-r[2]*q[3]+r[3]*q[2],
            r[0]*q[2]+r[1]*q[3]+r[2]*q[0]-r[3]*q[1],
            r[0]*q[3]-r[1]*q[2]+r[2]*q[1]+r[3]*q[0]]

def point_rotation_by_quarternion(point, q):
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    step1 = quarternion_mult(q, r)
    step2 = quarternion_mult(step1, q_conj)
    return step2[1:]

def factor(p, start=np.array([0.0, 0.0, 0.0]), clip=3.5):
    norm = np.linalg.norm(p - start)
    if norm > clip:
        return 0.0
    return np.exp(-0.5*norm)/10