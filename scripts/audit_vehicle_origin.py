"""Audit stored evidence before enabling planning from a vehicle reference point.

No inference of chassis identity from a frame name; no carving of unknown cells.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sitemind.fusion import resolve_frame_transform


def planar_direction(transform: np.ndarray) -> tuple[list[float], float]:
    direction = np.asarray(transform[:2, 0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError('frame x-axis has no planar direction')
    direction /= norm
    return direction.tolist(), float(np.degrees(np.arctan2(direction[1], direction[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('camera_root', type=Path)
    parser.add_argument('layer_root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/vehicle_origin_audit.json'))
    args = parser.parse_args()
    reports = []
    for path in sorted(args.camera_root.glob('frame_*/calibrated_frame.npz')):
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        tf_snapshot = [
            (item['parent'], item['child'], np.asarray(item['parent_from_child'], dtype=np.float64))
            for item in metadata.get('tf_snapshot', [])
        ]
        with np.load(path) as raw:
            cloud_from_camera = np.linalg.inv(raw['lidar_to_camera'])
            camera_origin = cloud_from_camera[:3, 3]
            camera_direction = cloud_from_camera[:3, 2]
            has_pose = 'world_from_pointcloud' in raw
        with np.load(args.layer_root / path.parent.name / 'fusion_layers.npz') as layers:
            resolution = float(layers['resolution_m'])
            row = int(np.floor(-float(layers['y_min_m']) / resolution))
            col = int(np.floor(-float(layers['x_min_m']) / resolution))
            inside = 0 <= row < layers['geometry_valid'].shape[0] and 0 <= col < layers['geometry_valid'].shape[1]
            observed = bool(layers['geometry_valid'][row, col]) if inside else False
            traversable = bool(observed and np.isfinite(layers['fused_risk'][row, col])
                               and layers['fused_risk'][row, col] < 0.99) if inside else False
        report = dict(
            frame=path.parent.name, pointcloud_frame=metadata['pointcloud_frame'],
            camera_origin_in_pointcloud_m=camera_origin.tolist(),
            camera_optical_axis_in_pointcloud=camera_direction.tolist(),
            has_world_pose=has_pose, has_tf_snapshot=bool(metadata.get('tf_snapshot')),
            coordinate_origin_cell=[row, col], origin_observed=observed,
            origin_traversable_without_override=traversable,
        )
        if tf_snapshot:
            try:
                base_frame = metadata['pointcloud_frame']
                base_from_chassis = resolve_frame_transform(tf_snapshot, 'e2_chassis', base_frame)
                base_from_upper = resolve_frame_transform(tf_snapshot, 'e2_upper_carriage', base_frame)
                chassis_forward, chassis_yaw = planar_direction(base_from_chassis)
                upper_forward, upper_yaw = planar_direction(base_from_upper)
                report.update(
                    chassis_reference_verified=True,
                    chassis_origin_in_pointcloud_m=base_from_chassis[:3, 3].tolist(),
                    chassis_forward_xy_in_pointcloud=chassis_forward,
                    chassis_yaw_deg_in_pointcloud=chassis_yaw,
                    upper_origin_in_pointcloud_m=base_from_upper[:3, 3].tolist(),
                    upper_forward_xy_in_pointcloud=upper_forward,
                    upper_yaw_deg_in_pointcloud=upper_yaw,
                    upper_minus_chassis_yaw_deg=float((upper_yaw - chassis_yaw + 180.) % 360. - 180.),
                )
            except ValueError as error:
                report.update(chassis_reference_verified=False, tf_resolution_error=str(error))
        else:
            report['chassis_reference_verified'] = False
        reports.append(report)
    if not reports:
        raise SystemExit('No calibrated frames found')
    summary = dict(
        frame_count=len(reports),
        world_pose_frames=sum(item['has_world_pose'] for item in reports),
        tf_snapshot_frames=sum(item['has_tf_snapshot'] for item in reports),
        origin_observed_frames=sum(item['origin_observed'] for item in reports),
        origin_traversable_frames=sum(item['origin_traversable_without_override'] for item in reports),
        vehicle_reference_verified=all(item['chassis_reference_verified'] for item in reports),
        vehicle_reference_frame='e2_chassis',
        current_exclusion_rectangle_m=dict(x_min=-3., x_max=4., y_min=-1.6, y_max=1.6),
        limitations=[
            'Exclusion rectangle is an implementation filter, not verified machine dimensions.',
            'e2_chassis is used as the physical chassis reference; this does not claim its origin is the geometric track centre.',
            'World pose alone does not establish the cab-to-undercarriage relationship.',
            'No free-space carving or substitute-start routing is authorized by this audit.',
        ], frames=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
