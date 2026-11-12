"""Estimate effective steering from logged bicycle-model curvature."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
L = 0.275
results = []
for path in sorted(ROOT.rglob('*.csv')):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows or 'vx_m_s' not in rows[0]:
        continue
    timestamps = np.array([int(r['stamp_sec']) * 10**9 + int(r['stamp_nanosec']) for r in rows])
    t = (timestamps - timestamps[0]) / 1e9
    v = np.array([float(r['vx_m_s']) for r in rows])
    w = np.array([float(r['wz_rad_s']) for r in rows])
    keep = (t > 3) & (np.abs(v) > 0.35)
    if not keep.any():
        continue
    if 'steering' in rows[0]:
        commands = {float(r['steering']) for r in rows if r['steering']}
        if len(commands) != 1:
            continue
        command = commands.pop()
    elif path.name == 'first.csv':
        command = 0.0  # User identified this earlier test as zero steering.
    else:
        continue
    angle = np.degrees(np.arctan(L * np.dot(v[keep], w[keep]) / np.dot(v[keep], v[keep])))
    instantaneous = np.degrees(np.arctan(L * w[keep] / v[keep]))
    lo, hi = np.percentile(instantaneous, [10, 90])
    results.append(dict(test=path.parent.name if path.name != 'first.csv' else 'zero (forward)', command=command, angle_deg=angle, p10_deg=lo, p90_deg=hi, samples=int(keep.sum())))

zero = next(r['angle_deg'] for r in results if r['command'] == 0)
left = np.mean([r['angle_deg'] for r in results if r['command'] > 0])
right = np.mean([r['angle_deg'] for r in results if r['command'] < 0])
right_u = np.mean([r['command'] for r in results if r['command'] < 0])
gain_left = left - zero
gain_right = (right - zero) / right_u
with (ROOT / 'steering_calibration_summary.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
for r in results:
    ax.errorbar(r['command'], r['angle_deg'], yerr=[[r['angle_deg']-r['p10_deg']], [r['p90_deg']-r['angle_deg']]], fmt='o', capsize=4, label=r['test'])
u = np.linspace(right_u, 1, 200)
a = zero + np.where(u >= 0, gain_left*u, gain_right*u)
ax.plot(u, a, color='black', label='Piecewise linear fit (equal weight per test)')
u_extra = np.linspace(-1, right_u, 40)
ax.plot(u_extra, zero + gain_right*u_extra, '--', color='gray', label='Untested extrapolation')
ax.axhline(0, color='gray', linewidth=.7)
ax.axvline(0, color='gray', linewidth=.7)
ax.set(xlabel='Normalized steering command u', ylabel='Effective steering angle (degrees)', title='Steering calibration; bars show sample 10th–90th percentiles', xlim=(-1.05, 1.05))
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.savefig(ROOT / 'steering_calibration.png', dpi=180)
print(f'offset={zero:.6f} deg; positive gain={gain_left:.6f} deg/unit; negative gain={gain_right:.6f} deg/unit')
print(f'zero angle command={-zero/gain_right:.6f}')
for r in results:
    print(r)
