import gzip
from itertools import islice
import re

import numpy as np
import torch


def _open_foam(path):
    if path.is_file():
        return path.open("rb")
    return gzip.open(path.with_suffix(path.suffix + ".gz"), "rb")


def _patches(path):
    with _open_foam(path) as stream:
        contents = stream.read().decode()
    contents = re.sub(r"/\*.*?\*/|//[^\n]*", "", contents, flags=re.S)
    return {
        name.strip('"'): dict(re.findall(r"\b(\w+)\s+([^;]+);", body))
        for name, body in re.findall(r'([^\s{}]+)\s*\{([^{}]*)\}', contents)
    }


def _mesh_list(stream):
    header = []
    for line in stream:
        if line.strip() == b"}":
            break
        header.append(line)
    if not re.search(rb"\bformat\s+ascii\s*;", b"".join(header)):
        raise ValueError("Boundary sampling requires ASCII OpenFOAM mesh lists")
    count = int(next(line for line in stream if line.strip()).strip())
    if next(line for line in stream if line.strip()).strip() != b"(":
        raise ValueError("Expected an OpenFOAM mesh list")
    return count


def _face_geometry(vertices):
    # Compute the area vector and centroid using triangles around the vertex mean.
    origin = vertices.mean(axis=0)
    offsets = vertices - origin
    following = np.roll(offsets, -1, axis=0)
    triangles = 0.5 * np.cross(offsets, following)
    area_vector = triangles.sum(axis=0)
    area = np.linalg.norm(area_vector)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Boundary face has invalid area")
    normal = area_vector / area
    weights = triangles @ normal
    center = origin + (weights @ (offsets + following)) / (3 * area)
    return np.r_[center, normal, area]


def _mesh_geometry(mesh, patches):
    with _open_foam(mesh / "points") as stream:
        count = _mesh_list(stream)
        rows = b"".join(islice(stream, count)).translate(None, b"()")
        points = np.fromstring(rows.decode(), sep=" ").reshape(count, 3)
    with _open_foam(mesh / "faces") as stream:
        count = _mesh_list(stream)
        index = 0
        for name, start, size, group, target in sorted(patches, key=lambda patch: patch[1]):
            if start < index or start + size > count:
                raise ValueError(f"Invalid face range for {name}")
            faces = islice(stream, start - index, start - index + size)
            geometry = np.empty((size, 7))
            for i in range(size):
                line = next(faces)
                n, labels = line.split(b"(", 1)
                vertices = np.fromstring(labels.split(b")", 1)[0].decode(), dtype=np.int64, sep=" ")
                if len(vertices) != int(n):
                    raise ValueError(f"Invalid face in {name}")
                geometry[i] = _face_geometry(points[vertices])
            index = start + size
            yield group, target, geometry


class VelocityBoundaries:
    """Equal weight per case and condition, with area-weighted faces in each group."""

    def __init__(self, cases, batch_size, seed, device):
        if batch_size <= 0:
            raise ValueError("Boundary sample size must be positive")
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = []
        max_speed = max(case["u_ref"] for case in cases)
        for case in cases:
            directory = case["path"].parent
            mesh = directory / "constant/polyMesh"
            conditions = _patches(directory / "0/U")
            patches = []
            for name, patch in _patches(mesh / "boundary").items():
                if "nFaces" not in patch or int(patch["nFaces"]) == 0:
                    continue
                condition = conditions.get(name)
                if condition is None:
                    raise ValueError(f"Missing velocity condition for {directory.name}/{name}")
                kind = condition["type"]
                is_wall = patch["type"] == "wall"
                if kind in {"slip", "symmetry", "symmetryPlane"}:
                    group, target = "slip", np.zeros(3)
                elif kind == "noSlip":
                    group, target = "wall", np.zeros(3)
                elif kind in {"fixedValue", "movingWallVelocity"}:
                    value = re.fullmatch(r"uniform\s*\(([^()]+)\)", condition.get("value", ""))
                    if value is None:
                        raise ValueError(f"Expected uniform velocity for {directory.name}/{name}")
                    target = np.fromstring(value[1], sep=" ")
                    if target.shape != (3,) or not np.isfinite(target).all():
                        raise ValueError(f"Invalid velocity for {directory.name}/{name}")
                    # These are steady cases with a static mesh.
                    group = "wall" if is_wall else "fixed"
                elif is_wall:
                    raise ValueError(f"Unsupported wall velocity condition {kind} at {directory.name}/{name}")
                else:
                    continue
                patches.append((name, int(patch["startFace"]), int(patch["nFaces"]), group, target))
            grouped = {}
            lengths = np.array([case["l_ref"], case["cross_ref"], case["cross_ref"]])
            speed = case["u_ref"]
            for group, target, geometry in _mesh_geometry(mesh, patches):
                center, normal, area = geometry[:, :3], geometry[:, 3:6], geometry[:, 6:]
                conditioning = np.full_like(area, speed / max_speed)
                rows = np.column_stack((center / lengths, conditioning, normal,
                                        np.broadcast_to(target / speed, center.shape), area))
                grouped.setdefault(group, []).append(rows)
            if not grouped:
                raise ValueError(f"No supported velocity boundaries in {directory}")
            for group, rows in grouped.items():
                faces = torch.as_tensor(np.concatenate(rows), dtype=torch.float32)
                cumulative_area = faces[:, -1].double().cumsum(dim=0)
                self.groups.append((group, faces, cumulative_area))
            summary = ", ".join(f"{group}={sum(len(r) for r in rows):,}" for group, rows in grouped.items())
            print(f"Velocity boundaries for {directory.name}: {summary}")

    def sample(self):
        batch = []
        for group, faces, cumulative_area in self.groups:
            draws = torch.rand(self.batch_size, dtype=torch.float64, generator=self.generator)
            indices = torch.searchsorted(cumulative_area, cumulative_area[-1] * draws)
            batch.append((group, faces[indices].to(self.device)))
        return batch

    @staticmethod
    def loss(model, batch):
        sizes = [len(faces) for _, faces in batch]
        predictions = model(torch.cat([faces[:, :4] for _, faces in batch]))[:, :3]
        losses = []
        for (group, faces), prediction in zip(batch, predictions.split(sizes)):
            error = prediction - faces[:, 7:10]
            if group == "slip":
                error = (error * faces[:, 4:7]).sum(dim=1)
                losses.append(error.square().mean())
            else:
                losses.append(error.square().sum(dim=1).mean())
        return torch.stack(losses).mean()
