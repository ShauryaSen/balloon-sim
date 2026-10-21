"""Turn one cBottle-video rollout into a global wind record this simulator reads

    python convert_cbottle.py ROLLOUT_DIR OUT.npz [--levels 200,50] [--step 2] [--lat-limit 80]

cBottle writes global winds on a HEALPix grid (49,152 pixels, about 100 km) every 6 hours, at the pressure
levels 1000, 850, 700, 500, 300, 200, 50 and 10 hPa. This script reads the chosen levels, interpolates each
frame onto a regular lat/lon grid covering the whole globe between the latitude limits, and writes the keys
`load_record` expects: u, v, hours, altitude_km, latitude, longitude. Longitude runs 0 to 360 - step; the
simulator wraps it. No forecast is written: the forecast is the truth under drawn noise (DESIGN.md, 5).

Runs on Unicorn in the cbottle env (needs xarray, healpy): source ~/envs/cbottle/bin/activate
"""

import argparse
import glob
import os

import healpy as hp
import numpy as np
import xarray as xr

ALTITUDE_KM = {1000: 0.1, 850: 1.5, 700: 3.0, 500: 5.5, 300: 9.2, 200: 11.8, 50: 20.6, 10: 31.0}
"""Standard-atmosphere height of each pressure level"""


def main() -> None:
    ask = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ask.add_argument("rollout_dir")
    ask.add_argument("out_npz")
    ask.add_argument("--levels", default="200,50", help="pressure levels in hPa, low altitude first")
    ask.add_argument("--step", type=float, default=2.0, help="grid spacing in degrees")
    ask.add_argument("--lat-limit", type=float, default=80.0, help="the grid covers -limit to +limit")
    asked = ask.parse_args()
    levels = [int(l) for l in asked.levels.split(",")]

    files = sorted(glob.glob(os.path.join(asked.rollout_dir, "*.nc")))
    ds = xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
    ds = ds.sortby("time")
    _, keep = np.unique(ds.time.values, return_index=True)  # chained clips can repeat a frame
    ds = ds.isel(time=keep)
    nest = "nest" in (str(ds.attrs) + str(ds["crs"].attrs if "crs" in ds else "")).lower()
    print(f"{ds.sizes['time']} frames, nside {hp.npix2nside(ds.sizes['pix'])}, {'NEST' if nest else 'RING'} order")

    lat = np.arange(-asked.lat_limit, asked.lat_limit + 1e-6, asked.step)
    lon = np.arange(0.0, 360.0 - 1e-6, asked.step)
    LON, LAT = np.meshgrid(lon, lat)
    theta, phi = np.radians(90.0 - LAT.ravel()), np.radians(LON.ravel())

    def on_grid(frames: np.ndarray) -> np.ndarray:  # (time, pix) -> (time, n_lat, n_lon)
        return np.stack([hp.get_interp_val(f, theta, phi, nest=nest).reshape(LAT.shape) for f in frames]).astype(np.float32)

    u = np.stack([on_grid(ds[f"U{l}"].values) for l in levels], axis=1)  # (time, alt, lat, lon)
    v = np.stack([on_grid(ds[f"V{l}"].values) for l in levels], axis=1)
    t = ds.time.values
    hours = ((t - t[0]) / np.timedelta64(1, "h")).astype(np.float32)
    altitude_km = np.asarray([ALTITUDE_KM[l] for l in levels], np.float32)
    order = np.argsort(altitude_km)  # ascending altitude, as load_record wants

    np.savez_compressed(
        asked.out_npz,
        latitude=lat.astype(np.float32),
        longitude=lon.astype(np.float32),
        hours=hours,
        altitude_km=altitude_km[order],
        pressure_hpa=np.asarray(levels, np.float32)[order],
        u=u[:, order],
        v=v[:, order],
        start_time=str(t[0])[:13],
    )
    print(f"wrote {asked.out_npz}: u/v {u.shape} (time, altitude, lat, lon), hours {hours[0]:.0f}..{hours[-1]:.0f}")
    for i in order:
        speed = np.hypot(u[:, i], v[:, i])
        print(f"  {levels[i]:>4} hPa, {altitude_km[i]:4.1f} km: mean speed {speed.mean():5.1f} m/s, max {speed.max():5.1f}")


if __name__ == "__main__":
    main()
