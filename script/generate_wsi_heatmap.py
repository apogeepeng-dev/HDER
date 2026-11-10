import os
import sys

# 作用：在部分 Windows/Conda 环境下避免 OpenMP 重复加载导致程序直接退出
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
try:
    import openslide
except ImportError:
    openslide = None
from PIL import Image

# 作用：在无图形界面/Qt 环境不完整时，使用无交互后端避免 Qt 插件报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 作用：确保从 script/ 目录运行时也能导入项目根目录下的 models 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


"""
生成 ViLa-MIL 的 WSI 注意力热图（patch-level）。

思路：
1) 从 h5 读取 patch 坐标 coords（以及可选的特征）。
2) 载入训练好的模型 checkpoint。
3) 用 cross-attention 得到 prototype->patch 的注意力分布。
4) 用 bag-level prototype 权重对其加权，得到每个 patch 的最终分数。
5) 将分数绘制到坐标平面并保存为 heatmap 图片。

注意：
- 本脚本适配你项目里的 ViLa_MIL / ViLa_MIL_v2 结构（含 cross_attention_1 + attention_V/U/weights）。
- 若使用 ViLa_MIL_3 并启用 GNN，可先关掉 GNN 可视化纯注意力，或自行扩展到空间增强后的特征。
"""


def load_h5_features_and_coords(h5_path):
    with h5py.File(h5_path, "r") as f:
        coords = np.array(f["coords"])
        features = None
        # 常见命名兼容：features / feats
        if "features" in f:
            features = np.array(f["features"])
        elif "feats" in f:
            features = np.array(f["feats"])

    if features is None:
        raise ValueError(f"在 {h5_path} 中未找到 features/feats 数据集。")

    return features, coords


def build_model(model_type, n_classes, text_prompt, prototype_number, num_heads=4, attn_dropout=0.1, abmil_dropout=False):
    import ml_collections

    config = ml_collections.ConfigDict()
    config.input_size = 1024
    config.hidden_size = 192
    config.text_prompt = text_prompt
    config.prototype_number = prototype_number
    config.num_heads = num_heads
    config.attn_dropout = attn_dropout
    config.abmil_dropout = abmil_dropout

    if model_type == "ViLa_MIL":
        from models.model_ViLa_MIL import ViLa_MIL_Model
        model = ViLa_MIL_Model(config=config, num_classes=n_classes)
    elif model_type == "ViLa_MIL_v2":
        from models.model_ViLa_MIL_2 import ViLa_MIL_Model
        model = ViLa_MIL_Model(config=config, num_classes=n_classes)
    elif model_type == "ViLa_MIL_3":
        from models.model_ViLa_MIL_3 import ViLa_MIL_Model_Improved
        # 可视化注意力时默认关闭 GNN，避免额外依赖/形状复杂度
        config.use_gnn = False
        config.lambda_div = 0.1
        config.spatial_k = 8
        config.gat_heads = 4
        model = ViLa_MIL_Model_Improved(config=config, num_classes=n_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model


@torch.no_grad()
def patch_attention_scores_vila(model, x):
    """
    基于 cross-attention + prototype聚合权重，输出 patch-level 分数。

    x: [N, D]
    return: scores [N]
    """
    device = next(model.parameters()).device
    x = x.to(device).float()

    # 兼容 query prototype 名称
    if hasattr(model, "learnable_image_center"):
        proto = model.learnable_image_center
        cross = model.cross_attention_1
    elif hasattr(model, "proto_low"):
        proto = model.proto_low
        cross = model.cross_attention_low
    else:
        raise RuntimeError("模型中未找到 prototype 参数（learnable_image_center/proto_low）。")

    # MultiheadAttention 输入形状按项目实现：query [K,1,D], key/value [N,D]
    comp, raw = cross(proto, x, x, need_weights=True, need_raw=True)

    # raw 期望 [1, heads, K, N]；先 softmax 到 patch 维
    if raw.dim() == 4:
        attn_proto2patch = F.softmax(raw.mean(dim=1).squeeze(0), dim=-1)  # [K, N]
    elif raw.dim() == 3:
        attn_proto2patch = F.softmax(raw.squeeze(0), dim=-1)  # [K, N]
    else:
        raise RuntimeError(f"unexpected raw attention shape: {tuple(raw.shape)}")

    # prototype 聚合权重 A_proto
    if hasattr(model, "bag_attention"):
        # ViLa_MIL_v2/v3 路径
        H = comp.squeeze(1).float()  # [K,D]
        A, _ = model.bag_attention(H)
        A = F.softmax(A.transpose(1, 0), dim=1).squeeze(0)  # [K]
    else:
        # 原始 ViLa_MIL 路径
        H = comp.squeeze().float()  # [K,D]
        A_V = model.attention_V(H)
        A_U = model.attention_U(H)
        A = model.attention_weights(A_V * A_U)
        A = F.softmax(A.transpose(1, 0), dim=1).squeeze(0)  # [K]

    # patch 分数：score_j = sum_k A_k * attn_{k,j}
    scores = torch.matmul(A, attn_proto2patch)  # [N]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores.detach().cpu().numpy()


def draw_heatmap(coords, scores, out_png, point_size=8, cmap="jet"):
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(10, 8), dpi=180)
    plt.scatter(x, y, c=scores, s=point_size, cmap=cmap, marker='s', linewidths=0)
    plt.colorbar(label="attention score")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("WSI Patch Attention Heatmap")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def generate_continuous_heatmap_overlay(
    coords,
    scores,
    wsi_image,
    out_png,
    patch_size=256,
    downsample_x=1.0,
    downsample_y=1.0,
    alpha=0.45,
    blur_ksize=31,
    cmap="jet",
):
    """
    标准 WSI 热力图流程：
    1) patch 分数回填到缩略图同尺寸 score map；
    2) 对 score map 平滑插值，得到连续热图；
    3) 颜色映射后与 WSI 底图半透明叠加。
    """
    base = np.array(wsi_image.convert("RGB"))
    h, w = base.shape[:2]

    score_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    pw = max(1, int(round(patch_size / float(downsample_x))))
    ph = max(1, int(round(patch_size / float(downsample_y))))

    for (x, y), s in zip(coords, scores):
        x0 = int(round(float(x) / float(downsample_x)))
        y0 = int(round(float(y) / float(downsample_y)))
        x1 = min(w, x0 + pw)
        y1 = min(h, y0 + ph)
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        score_map[y0:y1, x0:x1] += float(s)
        count_map[y0:y1, x0:x1] += 1.0

    valid_mask = count_map > 0
    score_map[valid_mask] = score_map[valid_mask] / count_map[valid_mask]

    # 插值/平滑：先高斯平滑，再归一化
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    smooth_map = cv2.GaussianBlur(score_map, (blur_ksize, blur_ksize), 0)

    if np.any(valid_mask):
        vmin = smooth_map[valid_mask].min()
        vmax = smooth_map[valid_mask].max()
        smooth_map[valid_mask] = (smooth_map[valid_mask] - vmin) / (vmax - vmin + 1e-8)

    # 颜色映射
    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = (cmap_fn(smooth_map)[..., :3] * 255).astype(np.uint8)

    # 仅在有效区域叠加，背景保持原图
    out = base.astype(np.float32).copy()
    heat_f = heat_rgb.astype(np.float32)
    out[valid_mask] = (1.0 - alpha) * out[valid_mask] + alpha * heat_f[valid_mask]
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)


def draw_wsi_overlay_heatmap(wsi_png_path, coords, scores, out_png, alpha=0.45, patch_size=256, downsample=1, cmap="jet"):
    """
    在 WSI 缩略图/导出图上叠加注意力热图。

    参数：
    - wsi_png_path: WSI 的底图（建议是与 coords 同坐标系的缩略图）
    - coords: patch 左上角坐标，原始坐标系
    - scores: 每个 patch 的注意力分数，范围 [0,1]
    - patch_size: 原始坐标系中的 patch 边长（如 256）
    - downsample: 底图相对于原始坐标系的缩放倍数（如底图是 1/32，则 downsample=32）
    """
    base = Image.open(wsi_png_path).convert("RGB")
    w, h = base.size

    heat = np.zeros((h, w), dtype=np.float32)
    norm = np.zeros((h, w), dtype=np.float32)

    patch_draw = max(1, int(round(patch_size / float(downsample))))

    for (x, y), s in zip(coords, scores):
        x0 = int(round(float(x) / float(downsample)))
        y0 = int(round(float(y) / float(downsample)))
        x1 = min(w, x0 + patch_draw)
        y1 = min(h, y0 + patch_draw)
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        heat[y0:y1, x0:x1] += float(s)
        norm[y0:y1, x0:x1] += 1.0

    mask = norm > 0
    heat[mask] = heat[mask] / norm[mask]
    if np.any(mask):
        vmin, vmax = heat[mask].min(), heat[mask].max()
        heat[mask] = (heat[mask] - vmin) / (vmax - vmin + 1e-8)

    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = (cmap_fn(heat)[..., :3] * 255).astype(np.uint8)

    base_np = np.array(base).astype(np.float32)
    heat_np = heat_rgb.astype(np.float32)

    out = base_np.copy()
    out[mask] = (1.0 - alpha) * base_np[mask] + alpha * heat_np[mask]
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)


def build_wsi_thumbnail_from_svs(wsi_path, thumbnail_max_side=4096):
    """
    从原始 WSI（如 .svs/.tiff）生成缩略图，并返回：
    - thumbnail(PIL.Image)
    - downsample_x, downsample_y（原始坐标 -> 缩略图坐标）
    """
    if openslide is None:
        raise ImportError("未安装 openslide-python，无法直接读取原始 WSI。请先安装 openslide-python 和 openslide-bin。")

    slide = openslide.OpenSlide(wsi_path)
    w0, h0 = slide.dimensions

    scale = float(max(w0, h0)) / float(thumbnail_max_side)
    scale = max(scale, 1.0)
    tw = int(round(w0 / scale))
    th = int(round(h0 / scale))

    thumb = slide.get_thumbnail((tw, th)).convert("RGB")

    downsample_x = float(w0) / float(tw)
    downsample_y = float(h0) / float(th)
    return thumb, downsample_x, downsample_y
def draw_binary_heatmap_overlay(coords, scores, wsi_image, out_png,
                                 threshold=0.5, patch_size=256,
                                 downsample_x=1.0, downsample_y=1.0,
                                 high_color=(220, 180, 60),   # 黄色
                                 low_color=(60, 30, 80),      # 深紫色
                                 bg_color=(100, 100, 100)):   # 灰色背景
    """
    复现 Figure 5 风格：patch 级二值热图
    - 高分 patch → 黄色（癌症）
    - 低分 patch → 深紫色
    - 无 patch 区域 → 灰色背景
    """
    base = wsi_image.convert("RGB")
    w, h = base.size

    # 初始化为灰色背景
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    pw = max(1, int(round(patch_size / float(downsample_x))))
    ph = max(1, int(round(patch_size / float(downsample_y))))

    for (x, y), s in zip(coords, scores):
        x0 = int(round(float(x) / float(downsample_x)))
        y0 = int(round(float(y) / float(downsample_y)))
        x1 = min(w, x0 + pw)
        y1 = min(h, y0 + ph)
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            continue
        x0, y0 = max(0, x0), max(0, y0)

        color = high_color if s >= threshold else low_color
        canvas[y0:y1, x0:x1] = color

    Image.fromarray(canvas).save(out_png)

def draw_wsi_overlay_heatmap_from_image(base_image, coords, scores, out_png, alpha=0.45, patch_size=256, downsample_x=1.0, downsample_y=1.0, cmap="jet"):
    """基于给定底图（PIL.Image）叠加热图。"""
    base = base_image.convert("RGB")
    w, h = base.size

    heat = np.zeros((h, w), dtype=np.float32)
    norm = np.zeros((h, w), dtype=np.float32)

    patch_w = max(1, int(round(patch_size / float(downsample_x))))
    patch_h = max(1, int(round(patch_size / float(downsample_y))))

    for (x, y), s in zip(coords, scores):
        x0 = int(round(float(x) / float(downsample_x)))
        y0 = int(round(float(y) / float(downsample_y)))
        x1 = min(w, x0 + patch_w)
        y1 = min(h, y0 + patch_h)
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            continue
        x0 = max(0, x0)
        y0 = max(0, y0)
        heat[y0:y1, x0:x1] += float(s)
        norm[y0:y1, x0:x1] += 1.0

    mask = norm > 0
    heat[mask] = heat[mask] / norm[mask]
    if np.any(mask):
        vmin, vmax = heat[mask].min(), heat[mask].max()
        heat[mask] = (heat[mask] - vmin) / (vmax - vmin + 1e-8)

    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = (cmap_fn(heat)[..., :3] * 255).astype(np.uint8)

    base_np = np.array(base).astype(np.float32)
    heat_np = heat_rgb.astype(np.float32)

    out = base_np.copy()
    out[mask] = (1.0 - alpha) * base_np[mask] + alpha * heat_np[mask]
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)
def draw_binary_heatmap_overlay(coords, scores, wsi_image, out_png,
                                 threshold=0.5, patch_size=256,
                                 downsample_x=1.0, downsample_y=1.0,
                                 high_color=(220, 180, 60),
                                 low_color=(60, 30, 80),
                                 bg_color=(120, 120, 120)):
    base = wsi_image.convert("RGB")
    w, h = base.size
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    pw = max(1, int(round(patch_size / float(downsample_x))))
    ph = max(1, int(round(patch_size / float(downsample_y))))
    for (x, y), s in zip(coords, scores):
        x0 = int(round(float(x) / float(downsample_x)))
        y0 = int(round(float(y) / float(downsample_y)))
        x1 = min(w, x0 + pw)
        y1 = min(h, y0 + ph)
        if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
            continue
        x0, y0 = max(0, x0), max(0, y0)
        color = high_color if s >= threshold else low_color
        canvas[y0:y1, x0:x1] = color
    Image.fromarray(canvas).save(out_png)

def main():
    parser = argparse.ArgumentParser("Generate WSI heatmap from trained ViLa-MIL model")
    parser.add_argument("--model_type", type=str, default="ViLa_MIL_v2", choices=["ViLa_MIL", "ViLa_MIL_v2", "ViLa_MIL_3"])
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练得到的 checkpoint 路径，如 s_0_checkpoint.pt")
    parser.add_argument("--feature_h5", type=str, required=True, help="某个 WSI 对应的 feature h5 文件（需包含 coords + features/feats）")
    parser.add_argument("--text_prompt_path", type=str, required=True)
    parser.add_argument("--prototype_number", type=int, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--out_png", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # 作用：可选在 WSI 底图上叠加热图（更直观看肿瘤区域与注意力重叠）
    parser.add_argument("--wsi_png", type=str, default=None, help="WSI 底图路径（png/jpg）")
    parser.add_argument("--wsi_path", type=str, default=None, help="原始 WSI 路径（svs/tif/tiff），与 wsi_png 二选一")
    parser.add_argument("--thumbnail_max_side", type=int, default=4096, help="从原始 WSI 导出缩略图时的最大边")
    parser.add_argument("--overlay_out_png", type=str, default=None, help="叠加图输出路径")
    parser.add_argument("--patch_size", type=int, default=256, help="原始坐标系下 patch size")
    parser.add_argument("--downsample", type=float, default=1.0, help="底图相对原始坐标的缩放倍数（仅 wsi_png 模式使用）")
    parser.add_argument("--alpha", type=float, default=0.45, help="热图叠加透明度")
    parser.add_argument("--blur_ksize", type=int, default=41, help="高斯平滑核大小（奇数）")

    # 可选参数
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--abmil_dropout", action="store_true", default=False)

    parser.add_argument("--binary_out_png", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()

    # 作用：与训练入口 main.py 保持一致，避免 np.loadtxt 在列数不齐时报错
    import pandas as pd
    text_prompt = np.array(pd.read_csv(args.text_prompt_path, header=None)).squeeze()

    model = build_model(
        model_type=args.model_type,
        n_classes=args.n_classes,
        text_prompt=text_prompt,
        prototype_number=args.prototype_number,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        abmil_dropout=args.abmil_dropout,
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    feats, coords = load_h5_features_and_coords(args.feature_h5)
    feats = torch.from_numpy(feats)

    scores = patch_attention_scores_vila(model, feats)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    draw_heatmap(coords, scores, args.out_png)
    print(f"Heatmap saved to: {args.out_png}")

    if args.wsi_png is not None or args.wsi_path is not None:
        overlay_out = args.overlay_out_png if args.overlay_out_png else args.out_png.replace('.png', '_overlay.png')
        os.makedirs(os.path.dirname(overlay_out), exist_ok=True)

        if args.wsi_path is not None:
            thumb, downsample_x, downsample_y = build_wsi_thumbnail_from_svs(
                args.wsi_path,
                thumbnail_max_side=args.thumbnail_max_side,
            )
            generate_continuous_heatmap_overlay(
                coords=coords,
                scores=scores,
                wsi_image=thumb,
                out_png=overlay_out,
                patch_size=args.patch_size,
                downsample_x=downsample_x,
                downsample_y=downsample_y,
                alpha=args.alpha,
                blur_ksize=args.blur_ksize,
                cmap="jet",
            )
            print(f"WSI overlay heatmap saved to: {overlay_out}")
            print(f"Auto downsample used: x={downsample_x:.4f}, y={downsample_y:.4f}")
        else:
            base_img = Image.open(args.wsi_png).convert("RGB")
            generate_continuous_heatmap_overlay(
                coords=coords,
                scores=scores,
                wsi_image=base_img,
                out_png=overlay_out,
                patch_size=args.patch_size,
                downsample_x=args.downsample,
                downsample_y=args.downsample,
                alpha=args.alpha,
                blur_ksize=args.blur_ksize,
                cmap="jet",
            )
            print(f"WSI overlay heatmap saved to: {overlay_out}")

    if args.binary_out_png and args.wsi_path is not None:
            os.makedirs(os.path.dirname(args.binary_out_png), exist_ok=True)
            thumb, dx, dy = build_wsi_thumbnail_from_svs(args.wsi_path, thumbnail_max_side=args.thumbnail_max_side)
            draw_binary_heatmap_overlay(coords, scores, thumb, args.binary_out_png, threshold=args.threshold, patch_size=args.patch_size, downsample_x=dx, downsample_y=dy)
            print(f"Binary heatmap saved to: {args.binary_out_png}")
if __name__ == "__main__":
    main()
