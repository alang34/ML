import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from config import params
import random


class OpenFOAMDataSet(Dataset):
    def __init__(self, csv_files, input_cols, target_cols, n_train, n_val, power=-2, seed=params.SEED, min_weight=0,
                 *, persistent_fraction=0.0, near_body_distance=0.10, wake_deficit=0.05,
                 supervise_vorticity=False):
        if not 0 <= persistent_fraction < 1:
            raise ValueError("Persistent training fraction must be in [0, 1)")
        self.u_ref_scale = max(float(case["u_ref"]) for case in params.CASE_FILES)
        data_list = []
        region_masks = []
        for csv in csv_files:
            csv = Path(csv)
            for case in params.CASE_FILES:
                if csv.resolve() == Path(case["path"]).resolve():
                    print(f"Match found! Adding u_ref to: {csv.name}")
                    df = pd.read_csv(csv)
                    df = df.rename(columns = params.COLUMN_MAP)
                    if persistent_fraction or supervise_vorticity:
                        if "wallDist" not in df or "wake_start_x" not in case:
                            raise ValueError(f"{csv.name} requires wallDist and wake_start_x for region sampling")
                        distance = df["wallDist"]
                        near_body = (distance > 0) & (distance <= near_body_distance)
                        wake = (distance > near_body_distance) & (df["x"] >= case["wake_start_x"])
                        wake &= df["u"] < (1 - wake_deficit) * case["u_ref"]
                        if "Block Name" in df:
                            interior = df["Block Name"].eq("internalMesh")
                            near_body &= interior
                            wake &= interior
                        region_masks.append((case["case_id"], near_body.to_numpy(), wake.to_numpy()))
                    if supervise_vorticity:
                        if "vorticity_0" not in df:
                            raise ValueError(f"{csv.name} requires vorticity_0 for wake supervision")
                        # Nondimensional streamwise vorticity is omega_x * L / Uref.
                        df["vorticity_0"] *= case["l_ref"] / case["u_ref"]
                    df["u_ref"] = case["u_ref"]
                    df["u_ref"] = df["u_ref"] / self.u_ref_scale
                    
                    df["x"] = df["x"]/case["l_ref"]
                    df["y"] = df["y"]/case["cross_ref"]
                    df["z"] = df["z"]/case["cross_ref"]
                    df["p"] = df["p"]/(float(case["u_ref"])**2)
                    
                    df["u"] = df["u"]/case["u_ref"]
                    df["v"] = df["v"]/case["u_ref"]
                    df["w"] = df["w"]/case["u_ref"]
                    data_list.append(df)
                    break
        if data_list:
            data_df = pd.concat(data_list, ignore_index=True)

            print("\n--- Success! ---")
            print("Final combined DataFrame shape:", data_df.shape)
            print("Columns:", data_df.columns.tolist())
        else:
            raise RuntimeError("No matching files were found.")
        self.input_cols = input_cols
        self.target_cols = target_cols
        missing_inputs = [col for col in input_cols if col not in data_df.columns]
        missing_targets = [col for col in target_cols if col not in data_df.columns]
        if missing_inputs:
            raise ValueError(f"Missing input columns: {missing_inputs}")
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        inputs = torch.tensor(data_df[input_cols].to_numpy(),dtype=torch.float32)

        targets = torch.tensor(data_df[target_cols].to_numpy(),dtype=torch.float32)

        print("Input columns:", input_cols)
        print("Target columns:", target_cols)
        print("Input tensor shape:", inputs.shape)
        print("Target tensor shape:", targets.shape)

        train_idx, val_idx = self.validationGen(inputs, targets, n_train, n_val, power=power, seed=seed, min_weight=min_weight)
        self._all_inputs, self._all_targets = inputs, targets
        self._crossflow_sampling_scores = self.getCrossflowSamplingScores(targets)
        self._val_mask = torch.zeros(len(targets), dtype=torch.bool)
        self._val_mask[val_idx] = True
        self._sampling_probabilities = self.samplingProbabilities(
            targets, power, min_weight, self._val_mask,
            crossflow_scores=self._crossflow_sampling_scores,
        )
        self._n_train, self._power, self._min_weight = n_train, power, min_weight
        reference_inputs, reference_targets = inputs[train_idx], targets[train_idx]
        self.training_bounds = reference_inputs.amin(dim=0), reference_inputs.amax(dim=0)
        self.target_scales = reference_targets.std(dim=0).clamp_min(1e-6)
        self.target_mean = reference_targets.mean(dim=0)
        self._persistent_train_idx = self._persistent_indices(region_masks, int(n_train * persistent_fraction), seed)
        self._refresh_excluded = self._val_mask.clone()
        self._refresh_excluded[self._persistent_train_idx] = True
        if self._persistent_train_idx.numel():
            # Keep the original weighted draw for the remaining initial samples.
            rotating = train_idx[~self._refresh_excluded[train_idx]]
            train_idx = torch.cat((self._persistent_train_idx, rotating[:n_train - len(self._persistent_train_idx)]))
        self._set_training(train_idx)
        self.val_inputs, self.val_targets = inputs[val_idx], targets[val_idx]

        self.val_deficit = self.getDeficit(self.val_targets)
        if supervise_vorticity:
            self._prepare_vorticity(
                region_masks, torch.as_tensor(data_df["vorticity_0"].to_numpy(), dtype=torch.float32), seed,
            )

    def __len__(self):
        return self.train_inputs.shape[0]

    def __getitem__(self, idx):
        return self.train_inputs[idx], self.train_targets[idx]

    def refresh_training(self, seed):
        idx = self.sampleIndexes(
            self._all_targets, self._n_train - len(self._persistent_train_idx), self._power,
            self._min_weight, self._refresh_excluded, torch.Generator().manual_seed(seed),
            probabilities=self._sampling_probabilities,
        )
        self._set_training(torch.cat((self._persistent_train_idx, idx)))

    def _set_training(self, idx):
        self.train_inputs, self.train_targets = self._all_inputs[idx], self._all_targets[idx]
        self.train_crossflow_scores = self._crossflow_sampling_scores[idx]

    def _prepare_vorticity(self, regions, labels, seed):
        self.vorticity_training = []
        self.vorticity_validation = []
        generator = torch.Generator().manual_seed(seed)
        offset = 0
        for case_id, _, wake in regions:
            wake_idx = torch.where(torch.from_numpy(wake))[0] + offset
            offset += len(wake)
            wake_idx = wake_idx[torch.isfinite(labels[wake_idx])]
            validation = wake_idx[self._val_mask[wake_idx]]
            eligible = wake_idx[~self._val_mask[wake_idx]]
            persistent = eligible[self._refresh_excluded[eligible]]
            train = persistent if len(persistent) else eligible
            if not len(train):
                raise ValueError(f"{case_id} has no finite, non-validation wake vorticity targets")
            train_targets = labels[train]
            scale = train_targets.square().mean().sqrt().clamp_min(1e-6)
            self.vorticity_training.append((self._all_inputs[train], train_targets, scale))
            validation = validation[torch.randperm(len(validation), generator=generator)[:256]]
            if len(validation):
                self.vorticity_validation.append((self._all_inputs[validation], labels[validation]))
            print(f"Wake vorticity {case_id}: {len(train):,} training, {len(validation):,} validation points")

    def _persistent_indices(self, regions, count, seed):
        if not count:
            return torch.empty(0, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        group_count = 2 * len(regions)
        quota, extra = divmod(count, group_count)
        selected = []
        offset = 0
        for case_index, (case_id, near_body, wake) in enumerate(regions):
            eligible = ~self._val_mask[offset:offset + len(near_body)]
            for region_index, (name, mask) in enumerate((("near body", near_body), ("wake", wake))):
                requested = quota + (2 * case_index + region_index < extra)
                candidates = torch.where(torch.from_numpy(mask) & eligible)[0]
                take = min(requested, len(candidates))
                sampled = candidates[torch.randperm(len(candidates), generator=generator)[:take]]
                selected.append(sampled + offset)
                print(f"Persistent training {case_id}, {name}: {take:,}/{requested:,} points")
            offset += len(near_body)
        return torch.cat(selected)

    @staticmethod
    def getDeficit(targets_normalized):
        u_over_uref = targets_normalized[:, 0]
        return torch.clamp(1.0 - u_over_uref, min=0.0, max=1.0)

    @staticmethod
    def getCrossflowSamplingScores(targets_normalized):
        values = targets_normalized[:, 1:3]
        if values.shape[0] == 0:
            return values.clone()
        if not torch.isfinite(values).all():
            raise ValueError("Crossflow velocities must be finite")
        raw = values.to(torch.float64)
        scores = torch.zeros_like(raw)
        nonzero = raw != 0
        scores[nonzero] = raw[nonzero].sign() * raw[nonzero].abs().log().abs()
        scale = scores.abs().amax(dim=0, keepdim=True)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (scores / scale).to(values.dtype)

    @staticmethod
    def validationGen(inputs, targets, numOfTrain, numOfVal, power, seed, min_weight):
        generator = torch.Generator().manual_seed(seed)
        deficit = OpenFOAMDataSet.getDeficit(targets)
        validation_indexes = OpenFOAMDataSet.sampleIndexes(deficit, numOfVal, power, min_weight, None, generator)
        mask = torch.zeros(len(deficit), dtype=bool)
        mask[validation_indexes] = True
        training_indexes = OpenFOAMDataSet.sampleIndexes(targets, numOfTrain, power, min_weight, mask, generator)
        return training_indexes, validation_indexes

    @staticmethod
    def samplingProbabilities(targets, power, min_weight, excluded, field_weights=None, crossflow_scores=None):
        eligible = torch.ones(len(targets), dtype=torch.bool, device=targets.device)
        if excluded is not None:
            eligible &= ~excluded
        if not eligible.any():
            raise ValueError("No eligible points available for sampling")
        if targets.ndim == 2:
            deficit = OpenFOAMDataSet.getDeficit(targets)
            if crossflow_scores is None:
                crossflow_scores = OpenFOAMDataSet.getCrossflowSamplingScores(targets)
            pressure = targets[:, 3].to(torch.float64)
            low, high = pressure[eligible].min(), pressure[eligible].max()
            span = high - low
            pressure = (pressure - low) / torch.where(span > 0, span, torch.ones_like(span))
            default_weights = params.SUPERVISED_SAMPLING_FIELD_WEIGHTS
        else:
            deficit = targets
            if crossflow_scores is None:
                raise ValueError("A deficit vector requires crossflow_scores")
            default_weights = params.SUPERVISED_SAMPLING_FIELD_WEIGHTS[:3]
        if crossflow_scores.shape != (len(targets), 2):
            raise ValueError("crossflow_scores must have shape (N, 2) in [v, w] order")
        columns = [deficit.to(torch.float64), *((crossflow_scores.to(torch.float64) + 1) / 2).unbind(1)]
        if targets.ndim == 2:
            columns.append(pressure)
        weights = torch.as_tensor(default_weights if field_weights is None else field_weights,
                                  dtype=torch.float64, device=targets.device)
        if (weights.shape != (len(columns),) or not torch.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("Field weights must be finite, nonnegative, and have a positive sum")
        weights = weights / weights.sum()
        edges = torch.linspace(0, 1, 1000, dtype=torch.float64, device=targets.device)
        probabilities = torch.zeros(len(targets), dtype=torch.float64, device=targets.device)
        for weight, column in zip(weights, columns):
            if not torch.isfinite(column).all():
                raise ValueError("Sampling scores must be finite")
            bins = (torch.bucketize(column.contiguous(), edges, right=True) - 1).clamp(0, len(edges) - 2)
            counts = torch.bincount(bins[eligible], minlength=len(edges) - 1)
            point_weights = counts[bins[eligible]].to(torch.float64).reciprocal()
            probabilities[eligible] += weight * point_weights / point_weights.sum()
        return probabilities / probabilities.sum()

    @staticmethod
    def sampleIndexes(deficit, count, power, min_weight, excluded, generator, crossflow_scores=None, probabilities=None):
        if deficit.ndim == 2 or crossflow_scores is not None or probabilities is not None:
            if count == 0:
                return torch.empty(0, dtype=torch.long, device=deficit.device)
            if probabilities is None:
                probabilities = OpenFOAMDataSet.samplingProbabilities(
                    deficit, power, min_weight, excluded, crossflow_scores=crossflow_scores,
                )
            if excluded is not None:
                probabilities = probabilities.masked_fill(excluded, 0)
            if count > (probabilities > 0).sum().item():
                raise ValueError("Sample count exceeds the number of eligible points")
            return torch.multinomial(probabilities, count, replacement=False, generator=generator)
        num_bins = 1000
        bins = np.linspace(0.0, 1.0, num_bins, endpoint = True)
        fractions = np.array([(bin+1)**power+min_weight for bin in bins])
        fractions /= fractions.sum()
        indexes = []
        for i in range(0, len(fractions)-1, 1):
            points_desired = int(np.ceil(count*fractions[i]))
            available = (deficit>=bins[i]) & (deficit<bins[i+1])
            if excluded is not None:
                available &= ~excluded
            valid_points = torch.where(available)[0]
            shuffled = torch.randperm(len(valid_points), generator=generator)
            selected_points = valid_points[shuffled[:points_desired]]
            indexes.append(selected_points)
        indexes = torch.cat(indexes)
        return OpenFOAMDataSet.checkNumericalExcess(indexes, count, deficit, excluded, generator)
    
    @staticmethod
    def checkNumericalExcess(indexArr, target, deficit, mask, generator):
        n = len(indexArr) - target
        length = len(indexArr)
        if n > 0:
            perm = torch.randperm(length, generator=generator)
            indexArr = indexArr[perm[:target]]
        elif n < 0:
            used = torch.zeros(len(deficit),dtype=torch.bool)
            used[indexArr] = True
            if mask is not None:
                used |= mask
            available_idx = torch.where((~used) & (deficit<0.99))[0]
            perm = torch.randperm(len(available_idx), generator=generator)
            extra_idx = available_idx[perm[:-n]]
            indexArr = torch.cat([indexArr,extra_idx])

        return indexArr
