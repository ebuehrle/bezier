import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
import argparse

def bezier(t, ctrl):
    t = np.array(t)
    ctrl = np.array(ctrl)
    
    T, = t.shape
    n, d = ctrl.shape

    def casteljau(t, M):
        assert n == M[0].shape[0], "The second dimension of M does not match the number of control points."
        
        t = np.expand_dims(t, axis=(0, 2))
        for i in range(1, n):
            M.append((1 - t) * M[i-1][0:n-i, :, :] + t * M[i-1][1:n-i+1, :, :])
    
    M = [np.repeat(np.expand_dims(ctrl, axis=1), T, axis=1)]
    casteljau(np.array(t), M)
    return M

def test_single_time_single_point():
    assert equal(bezier((0,), [(0,0)]), [
        np.zeros((1,1,2)),
    ])

def test_single_time_mult_points():
    assert equal(bezier((1,), [(0,0), (1,1)]), [
        np.array([[[0,0]],[[1,1]]]),
        np.array([[[1,1]]]),
    ])

def test_mult_time_single_point():
    assert equal(bezier((0,0.13,0.5,1), [(0,0)]), [
        np.zeros((1,4,2)),
    ])

def test_mult_time_mult_points():
    assert equal(bezier((0,0.13,1), [(0,0), (1,1)]), [
        np.array([[[0,0],[0,0],[0,0]],[[1,1],[1,1],[1,1]]]),
        np.array([[[0,0],[0.13,0.13],[1,1]]]),
    ])

def equal(l1, l2):
    for np1, np2 in zip(l1, l2):
        if not np.array_equal(np1, np2):
            return False
    return True

def animate_bezier(t, M, frames_dir):
    assert len(t) == M[0].shape[1]
    for i, _ in tqdm(enumerate(t)):
        draw_bezier_frame(t, M, i, frames_dir)

def draw_bezier_frame(t, M, idx, frames_dir):
    assert len(M) == M[0].shape[0]
    assert len(t) == M[0].shape[1]
    assert 0 <= idx < len(t), f"idx must lie in [0, {len(t)-1}]"
    
    n, T, d = M[0].shape

    """Draw lines"""
    for i in range(n):
        color = (
            np.interp(i, [0, n-1], [199/255, 232/255]),
            np.interp(i, [0, n-1], [234/255, 25/255]),
            np.interp(i, [0, n-1], [253/255, 139/255]),
            1
        )
        linewidth=np.interp(i, [0, n-1], [1, 2.5])
        plt.plot(M[i][:n-i, idx, 0], M[i][:n-i, idx, 1], color=color, linewidth=linewidth, marker='x')

    """Draw sub-curves"""
    for i in range(n):
        color = (
            np.interp(i, [0, n-1], [199/255, 232/255]),
            np.interp(i, [0, n-1], [234/255, 25/255]),
            np.interp(i, [0, n-1], [253/255, 139/255]),
            1
        )
        linewidth=np.interp(i, [0, n-1], [1, 2.5])
        for j in range(n-i):
            plt.plot(M[i][j, :idx+1, 0], M[i][j, :idx+1, 1], color=color, linewidth=linewidth)

    plt.title(f"t = {t[idx]:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.savefig(os.path.join(frames_dir, f"{idx:06d}.png"))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Bézier curves and animations.')
    parser.add_argument('--frames-dir', default='frames/')
    args = parser.parse_args()

    ctrl = []
    while True:
        """Read in control points"""
        line = input().strip()
        if not line:
            break
        point = (float(p.strip()) for p in line.split(','))
        ctrl.append(tuple(point))
    
    t_step = float(input().strip())
    t = np.arange(0, 1+t_step, t_step)

    M = bezier(t, ctrl)

    if os.path.exists(args.frames_dir):
        shutil.rmtree(args.frames_dir)
    os.makedirs(args.frames_dir)
    animate_bezier(t, M, args.frames_dir)
