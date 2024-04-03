import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta=0.005):
    total_delta = (target_pt - start_pt)
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = (np.linalg.norm(total_delta) % max_delta)
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps
    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)
    return gen_waypoint, int(num_steps)

# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = R.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02 or num_steps < 2:
        def gen_ori(i):
            return initial_euler
    else:
        slerp = Slerp([1,num_steps], ori_chg)
        def gen_ori(i): 
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler
    return gen_ori
