"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)#PatchMaker负责把特征图切成局部 Patch，以及将 Patch 级异常分数重新组织起来。

        self.forward_modules = torch.nn.ModuleDict({})#创建一个空的、专门存放 PyTorch 网络层的字典容器，键是字符串名字，值是 nn.Module（网络模块）。

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(#从指定的骨干网络层提取中间特征
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator#把特征聚合模块加入 forward_modules 字典中。

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing#把预处理模块加入 forward_modules 字典中。

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator#把预适应聚合模块加入 forward_modules 字典中。
        #不同骨干网络层的特征通道数可能不同，空间尺寸也可能不同。如果直接把它们当成完全相同格式的向量进行比较，就会遇到维度不一致的问题。因此，需要先处理processing、再统一aggregator。
        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(#利用近邻距离计算异常程度。
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(#把 Patch 级的异常分数处理成与图片尺寸对应的异常图。不这样处理的话，异常分数就无法直观地反映整张图像的异常程度（mask_gt）。
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler#特征采样器，用于在训练和预测时对特征进行采样。

    def embed(self, data):#如果是DataLoader，则遍历每个批次然后逐批掉调用_embed方法。如果是一批图像则直接调用_embed方法。
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):#把特征图转换为 Patch 嵌入向量
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]#将特征从 GPU 移到 CPU 并转换为 NumPy 数组。
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)#features最初是特征提取模块输出的结果

        features = [features[layer] for layer in self.layers_to_extract_from]#把结果整理成一个列表，列表中只保留我们指定的网络层。

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features#将每个特征图切分为 Patch，并返回空间信息。返回空间信息就是特征图高度方向 patch 数量、宽度方向 patch 数量
        ]
        patch_shapes = [x[1] for x in features]#提取每个层的空间信息，即高度和宽度方向的 patch 数量。
        features = [x[0] for x in features]#提取每个层的特征图，去掉空间信息。
        ref_num_patches = patch_shapes[0]#第0个特征层的 patch 网格尺寸当作参考基准。

        for i in range(1, len(features)):
            _features = features[i]#取出第 i 个特征层的特征图。
            patch_dims = patch_shapes[i]#第 i 个特征层的 patch 网格尺寸。

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]#将特征图重塑形状,以便后续进行插值操作。[B, H*W, C, k, k]变成 [B, H, W, C, k, k]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)#[B, H, W, C, k, k]变成 [B, C, k, k, H, W]
            perm_base_shape = _features.shape#保存当前特征图的形状，以便后续恢复。
            _features = _features.reshape(-1, *_features.shape[-2:])#将特征图展平为 (B*C*k*k, H，W) 的形式，以便进行插值操作。
            #用双线性插值，把所有层的 patch 特征统一缩放到同一个参考网格`ref_num_patches`,对齐空间位置，之后才能把多层特征拼接到一起，形成单个空间位置的特征向量，最后存入 memory bank。
            _features = F.interpolate(#函数F.interpolate只接受（B，C，H，W）形式的输入，因此需要添加通道维度。所以前面的维度转换还有还原会存在。
                _features.unsqueeze(1),#在特征图前面添加一个通道维度
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",#使用双线性插值方法进行上采样。
                align_corners=False,
            )
            _features = _features.squeeze(1)#移除之前添加的通道维度。
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])#恢复成对齐后的原始特征图的形状。
            features[i] = _features#将处理后的特征图存回原始特征列表。
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]#批次维度和 Patch 网格中的位置合并

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:#如果提供 patch 网格尺寸，则返回处理后的特征图和 patch 网格尺寸。
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)#将输入图像转换为浮点类型并移动到指定设备。
                return self._embed(input_image)#调用内部嵌入函数，生成特征图。

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):#如果输入是字典类型，则提取其中的图像部分。
                    image = image["image"]
                features.append(_image_to_features(image))#列表features中存放的是每一批图片对应的局部特征
        '''
        features = np.concatenate(features, axis=0)#将每一批图片对应的局部特征拼接成一个大的特征数组。
        features = self.featuresampler.run(features)#对拼接后的特征数组进行采样，得到最终的特征表示。

        self.anomaly_scorer.fit(detection_features=[features])#真正建立memory bank的位置
        '''
        features = np.concatenate(features, axis=0)# Coreset 采样前：全部特征向量数量
        num_before = features.shape[0]# 执行 Coreset 采样
        features = self.featuresampler.run(features)# Coreset 采样后：Memory Bank 实际保留的特征数量
        num_after = features.shape[0]
        print(f"[Coreset] Before: {num_before}, "f"After: {num_after}, "f"Actual ratio: {num_after / num_before:.4f}, "f"Memory Bank shape: {features.shape}")
        self.anomaly_scorer.fit(detection_features=[features])
    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):#根据输入类型，决定是处理整个 DataLoader，还是直接处理一批图片。
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)#调用内部预测函数，生成异常分数和异常热力图。
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt#返回异常分数、异常热力(像素级异常分数)图、真实标签和像素级标注。

    def _predict(self, images):#测试图片如何变成异常分数和异常热力图
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]#将特征交给最近邻评分组件，得到异常分数；外面的 [0] 则取返回结果中的第一个元素。
            image_scores = self.patch_maker.unpatch_scores(#将每个 patch 的异常分数聚合为整张图像的异常分数。
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)#将后续维度整理成适合聚合的结构。
            image_scores = self.patch_maker.score(image_scores)#对每张图片的 Patch 分数进行最大值聚合。

            patch_scores = self.patch_maker.unpatch_scores(#回到分支，用来生成像素分割图
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]#获取第一个 patch 的形状，用于后续重塑异常分数。
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])#把一维 patch 分数，恢复回二维空间网格[B, H, W]现在得到的是**低分辨率异常分数图

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)#将低分辨率异常分数图转换（双线性插值变成原图尺寸然后归一化）得到与原始图像相同分辨率的异常分割图。

        return [score for score in image_scores], [mask for mask in masks]#返回整张图像的异常分数和像素级异常分割图。

    @staticmethod#静态方法，不需要实例化对象即可调用。
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")#返回拼接的的文件路径。

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(#保存FAISS 记忆库 Memory Bank
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {#用 pickle 保存 PatchCore 的超参配置字典
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:#'wb' 表示以二进制写模式打开文件。
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)#pickle.dump() 将 data 对象序列化并写入patchcore_params.pkl文件。
    #从保存的文件夹里，读取`patchcore_params.pkl`配置，重建 PatchCore 实例，再加载 FAISS 记忆库。
    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:#`-> None`：代表这个函数执行结束后，没有返回值。
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:#'rb' 表示以二进制读模式打开文件。
            patchcore_params = pickle.load(load_file)#pickle.load() 从文件中读取序列化对象并反序列化为 Python 对象。
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        # Older saved files use "scorer"; load() expects "score".
        if "anomaly_scorer_num_nn" in patchcore_params:
            patchcore_params["anomaly_score_num_nn"] = patchcore_params.pop(
                "anomaly_scorer_num_nn"
            )
        self.load(**patchcore_params, device=device, nn_method=nn_method)#加载 PatchCore 实例。

        self.anomaly_scorer.load(load_path, prepend)#加载 FAISS 记忆库。


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)#滑动窗口之后输出 patch 网格的高宽和输入特征图高宽保持不变。
        unfolder = torch.nn.Unfold(#使用 torch.nn.Unfold 将输入特征图展开为 patch 序列。
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (#得到patch的总数
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )#[B,C,P,P,L]
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)#[B,L,C,P,P]，L 表示 patch 数量。P 表示 patch 的边长。

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):#将 patch 分数重新排列回原始空间布局。
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)#将输入的 numpy 数组转换为 PyTorch 张量。
        while x.ndim > 1:#当输入张量维度大于 1 时，沿最后一个维度取最大值，逐步聚合 Patch 分数。
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
