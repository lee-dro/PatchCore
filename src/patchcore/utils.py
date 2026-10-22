import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if not len(image_paths):
        return
    masks_provided = mask_paths is not None
    if mask_paths is None:
        mask_paths = [None for _ in image_paths]
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        image = np.asarray(image)
        image_show = image.transpose(1, 2, 0) if image.shape[0] in (1, 3) else image
        mask = np.zeros(image_show.shape[:2])

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                mask = np.asarray(mask)

        savename = os.path.normpath(image_path)
        savename = "_".join(savename.split(os.sep)[-save_depth:])
        savename = os.path.splitext(savename)[0] + "_visualization.png"
        savename = os.path.join(savefolder, savename)
        mask_show = (
            mask.transpose(1, 2, 0)
            if mask.ndim == 3 and mask.shape[0] in (1, 3)
            else mask
        )
        mask_show = mask_show.squeeze()
        heatmap = np.asarray(segmentation).squeeze()  # 创建四个可视化面板
        f, axes = plt.subplots(1, 4, figsize=(12, 3))  # 1. 原始图像
        axes[0].imshow(image_show)
        axes[0].set_title("Original Image")  # 2. 数据集提供的真实异常标注
        axes[1].imshow(mask_show)
        axes[1].set_title("Ground Truth")  # 3. PatchCore 预测的异常热力图
        axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[2].set_title("Anomaly Heatmap")  # 4. 把热力图叠加到原图上
        axes[3].imshow(image_show)
        axes[3].imshow(heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[3].set_title("Overlay")  # 隐藏坐标轴，整理布局
        for ax in axes:
            ax.axis("off")

        f.tight_layout()
        LOGGER.debug("Saving segmentation: %s", savename)
        f.savefig(savename)
        plt.close(f)


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
    row_groups=None,
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    if row_groups is None:
        row_groups = ["all"] * len(results)
    if len(row_groups) != len(results):
        raise ValueError("Each result requires one method group.")
    group_names = list(dict.fromkeys(row_groups))
    group_means = {}
    for group in group_names:
        selected = [row for row, label in zip(results, row_groups) if label == group]
        group_means[group] = np.mean(selected, axis=0).tolist()
        for key, value in zip(column_names, group_means[group]):
            prefix = "" if len(group_names) == 1 else f"[{group}] "
            LOGGER.info("%s%s: %.3f", prefix, key, value)

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        for group, means in group_means.items():
            label = "Mean" if len(group_names) == 1 else f"Mean__{group}"
            csv_writer.writerow(([label] if row_names is not None else []) + means)

    mean_metrics = {
        (f"mean_{key}" if len(group_names) == 1 else f"mean_{group}_{key}"): value
        for group, values in group_means.items()
        for key, value in zip(column_names, values)
    }
    return mean_metrics
