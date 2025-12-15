# 作业总结
## 高斯滤波、LoG 检测、Canny 检测
本次作业围绕图像边缘检测展开，分为基础核心任务与拓展任务，依托 Lena 图像（灰度 / 彩色）和 BSDS500 数据集，系统考察高斯滤波、LoG 检测、Canny 检测的原理实现与性能评估，具体总结如下：

#### 一、核心任务：基础边缘检测（任务一、二）

<img width="1794" height="836" alt="image" src="https://github.com/user-attachments/assets/6f62b2a8-7ad6-4a9e-84ad-8a5f6ce423b8" />



1. **任务一：LoG 滤波与边缘检测**

   核心目标是基于 Lena 灰度 / 彩色图，完成高斯滤波与 LoG 检测的多参数实验：

   - 高斯滤波：需构造不同窗口大小（3×3 至 11×11）、不同尺度因子 σ（1、3、5、7）的高斯核，对图像（彩色图分通道）进行卷积平滑，记录不同参数下的滤波效果；
   - LoG 检测：基于高斯核推导 LoG 核，通过卷积实现边缘检测，计算边缘强度（卷积结果绝对值），可选实现零交叉检测，核心分析 σ 和窗口大小的影响（σ 越大，平滑越强、边缘越少越粗）。
   
   既然我们有 **lena512.bmp** 文件，这是一个标准的图像格式，我们可以直接用 OpenCV 来加载它作为灰度图，这比加载 .mat 文件更简单，并且可以移除对 scipy 库的依赖。

### ✅ 任务一要求实现情况总结

| **任务要求细节**                                             | **代码实现情况** | **对应的代码部分**                                           |
| ------------------------------------------------------------ | ---------------- | ------------------------------------------------------------ |
| **滤波器：高斯滤波核**                                       | **已实现**       | `apply_gaussian_filter` 函数使用 `cv2.GaussianBlur`。        |
| **边缘检测子：高斯拉普拉斯检测子 (LoG)**                     | **已实现**       | `apply_log_detection` 函数在高斯滤波后使用 `cv2.Laplacian`。 |
| **计算出边缘强度**                                           | **已实现**       | `apply_log_detection` 计算拉普拉斯结果的 **绝对值** (`np.abs()`) 并进行归一化。 |
| **数据：Lena 灰度图**                                        | **已实现**       | 代码加载 `lena/lena512.bmp` 并进行处理 (`# 5. 灰度图 LoG 处理`)。 |
| **数据：Lena 彩色图**                                        | **已实现**       | 代码加载 `lena/lena512color.tiff` 并进行 **分通道处理** (`# 6. 彩色图 LoG 处理`)。 |
| **窗口大小：从 $3 \times 3$ 到 $11 \times 11$，以 2 为间隔** | **已实现**       | 列表 `WINDOWS = [3, 5, 7, 9, 11]` 被遍历。                   |
| **尺度因子：从 1 像素到 7 像素，以 2 为间隔**                | **已实现**       | 列表 `SIGMAS = [1, 3, 5, 7]` 被遍历。                        |
| **结果展示：灰度图和彩色图连续显示**                         | **已实现**       | 最后的 `# 7. 结果示例展示` 部分创建了 $2 \times 3$ 的子图，同时展示了灰度和彩色的对比结果。 |

   - #### 我们可以看到任务一要求我们最后生成5✖️4✖️2（五种窗口大小*四种尺度因子*两种图（灰度和彩色））
代码如下：


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 设置参数列表 ---
SIGMAS = [1, 3, 5, 7]  # 尺度因子 sigma
WINDOWS = [3, 5, 7, 9, 11]  # 窗口大小 W x W (必须为奇数)

# --- 2. 文件路径设置 ---
# 假设你的数据在 'lena/' 文件夹中
IMAGE_BMP_PATH = 'lena/lena512.bmp'
IMAGE_TIFF_PATH = 'lena/lena512color.tiff'
RESULTS_DIR = 'results'

# 确保结果文件夹存在
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 3. 图像加载 ---
print("--- 图像加载 ---")
try:
    # 3.1 加载彩色图 (TIFF)
    # OpenCV 默认加载为 BGR 格式
    img_color = cv2.imread(IMAGE_TIFF_PATH)

    # 3.2 加载灰度图 (BMP)
    img_gray = cv2.imread(IMAGE_BMP_PATH, cv2.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        raise FileNotFoundError("无法找到或加载图片。请检查路径是否正确。")

except FileNotFoundError as e:
    print(f"致命错误: {e}")
    print("请检查 'lena/lena512.bmp' 和 'lena/lena512color.tiff' 文件路径。")
    exit()

# BGR 转 RGB，用于 Matplotlib 显示 (可选)
img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
print(f"图像加载成功。彩色图尺寸: {img_color.shape}，灰度图尺寸: {img_gray.shape}。")


# --- 4. 滤波与检测函数定义 ---

def apply_gaussian_filter(image, window_size, sigma):
    """应用高斯滤波：cv2.GaussianBlur"""
    kernel_size = (window_size, window_size)
    # sigmaX 和 sigmaY 均设置为 sigma
    filtered_img = cv2.GaussianBlur(image, kernel_size, sigma, sigma)
    return filtered_img


def apply_log_detection(image, window_size, sigma):
    """实现 LoG 边缘检测：高斯平滑 -> 拉普拉斯运算 -> 计算边缘强度"""

    # 1. 高斯平滑
    smoothed_img = apply_gaussian_filter(image, window_size, sigma)

    # 2. 拉普拉斯运算 (使用 64 位浮点数避免溢出)
    laplacian = cv2.Laplacian(smoothed_img, cv2.CV_64F)

    # 3. 计算边缘强度 (即绝对值)
    edge_strength = np.abs(laplacian)

    # 4. 标准化结果到 0-255 范围 (用于保存)
    max_val = np.max(edge_strength)
    if max_val > 0:
        edge_strength_normalized = (edge_strength / max_val) * 255
    else:
        edge_strength_normalized = np.zeros_like(edge_strength)

    # 转换为 8 位整型图像
    edge_strength_uint8 = np.uint8(edge_strength_normalized)

    return edge_strength_uint8


# --- 5. 灰度图处理循环 ---

print("\n--- 5. 灰度图 LoG 处理 ---")
for W in WINDOWS:
    for sigma in SIGMAS:
        try:
            log_edge_gray = apply_log_detection(img_gray, W, sigma)

            filename = f'gray_W{W}_s{sigma}.png'
            filepath = os.path.join(RESULTS_DIR, filename)
            cv2.imwrite(filepath, log_edge_gray)

            print(f" [成功] 灰度图 W={W}, sigma={sigma} -> {filename}")
        except Exception as e:
            print(f" [失败] 灰度图 W={W}, sigma={sigma} 出现错误: {e}")

# --- 6. 彩色图处理循环 ---

print("\n--- 6. 彩色图 LoG 处理 (分通道) ---")
# 分离 BGR 三个通道
B, G, R = cv2.split(img_color)

for W in WINDOWS:
    for sigma in SIGMAS:
        try:
            # 对每个通道应用 LoG
            log_B = apply_log_detection(B, W, sigma)
            log_G = apply_log_detection(G, W, sigma)
            log_R = apply_log_detection(R, W, sigma)

            # 合并结果
            log_color_combined = cv2.merge([log_B, log_G, log_R])

            # 保存结果图
            filename = f'color_W{W}_s{sigma}.png'
            filepath = os.path.join(RESULTS_DIR, filename)
            cv2.imwrite(filepath, log_color_combined)

            print(f" [成功] 彩色图 W={W}, sigma={sigma} -> {filename}")
        except Exception as e:
            print(f" [失败] 彩色图 W={W}, sigma={sigma} 出现错误: {e}")

# --- 7. 结果示例展示 (灰度图 + 彩色图联合比较) ---

print("\n--- 7. 结果示例展示 (Matplotlib 联合展示) ---")

W_small, sigma_small = 3, 1
W_large, sigma_large = 11, 7

# ----------------- 加载灰度图结果 -----------------
result_gray_small = cv2.imread(os.path.join(RESULTS_DIR, f'gray_W{W_small}_s{sigma_small}.png'), cv2.IMREAD_GRAYSCALE)
result_gray_large = cv2.imread(os.path.join(RESULTS_DIR, f'gray_W{W_large}_s{sigma_large}.png'), cv2.IMREAD_GRAYSCALE)

# ----------------- 加载彩色图结果 -----------------
# 注意：cv2.imread 加载的是 BGR 格式
result_color_small_bgr = cv2.imread(os.path.join(RESULTS_DIR, f'color_W{W_small}_s{sigma_small}.png'))
result_color_large_bgr = cv2.imread(os.path.join(RESULTS_DIR, f'color_W{W_large}_s{sigma_large}.png'))

# 检查是否成功加载了所有图片
if (result_gray_small is not None and result_gray_large is not None and
        result_color_small_bgr is not None and result_color_large_bgr is not None):

    # 转换彩色图结果的 BGR 为 RGB 格式
    result_color_small_rgb = cv2.cvtColor(result_color_small_bgr, cv2.COLOR_BGR2RGB)
    result_color_large_rgb = cv2.cvtColor(result_color_large_bgr, cv2.COLOR_BGR2RGB)

    # ----------------- 创建 2x3 子图布局 -----------------
    plt.figure(figsize=(12, 5))  # 调整窗口大小以便容纳 6 个图

    # --- 第一行: 灰度图结果 ---

    # 1. 原始灰度图 (img_gray 是在文件加载时生成的)
    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('1. Original Gray Image')
    plt.axis('off')

    # 2. 灰度图 LoG (细节/小尺度)
    plt.subplot(2, 3, 2)
    plt.imshow(result_gray_small, cmap='gray')
    plt.title(f'2. Gray LoG (W={W_small}, sigma={sigma_small}) - Detail')
    plt.axis('off')

    # 3. 灰度图 LoG (平滑/大尺度)
    plt.subplot(2, 3, 3)
    plt.imshow(result_gray_large, cmap='gray')
    plt.title(f'3. Gray LoG (W={W_large}, sigma={sigma_large}) - Smooth')
    plt.axis('off')

    # --- 第二行: 彩色图结果 ---

    # 4. 原始彩色图 (img_color_rgb 是在文件加载时生成的)
    plt.subplot(2, 3, 4)
    plt.imshow(img_color_rgb)
    plt.title('4. Original Color Image')
    plt.axis('off')

    # 5. 彩色图 LoG (细节/小尺度)
    plt.subplot(2, 3, 5)
    plt.imshow(result_color_small_rgb)
    plt.title(f'5. Color LoG (W={W_small}, sigma={sigma_small}) - Detail')
    plt.axis('off')

    # 6. 彩色图 LoG (平滑/大尺度)
    plt.subplot(2, 3, 6)
    plt.imshow(result_color_large_rgb)
    plt.title(f'6. Color LoG (W={W_large}, sigma={sigma_large}) - Smooth')
    plt.axis('off')

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

else:
    print("致命错误: 无法加载所有结果图进行联合展示。请检查 'results' 文件夹中的文件是否完整。")

print("\n所有任务一的处理和灰度/彩色联合结果展示已完成。")
```









2. **任务二：Canny 边缘检测**

<img width="2235" height="796" alt="image" src="https://github.com/user-attachments/assets/0750e815-e385-4683-944a-a64ec1d1b4a1" />

   聚焦 Canny 算法的应用与原理理解：

   - 实践层面：调用 OpenCV/scikit-image（Python）或 MATLAB 内置 Canny 函数，调整高低阈值（\(T_{high}\)/\(T_{low}\)）开展对比实验，分析阈值对边缘连接、噪声抑制的影响；
   - 原理层面：掌握 Canny 四步流程（高斯滤波→梯度计算→非极大值抑制→双阈值边缘跟踪），并在报告中详细阐述算法原理与实现流程。

### ✅ 任务二：Canny 边缘检测子 实现情况总结

| **任务要求细节**                           | **代码实现情况**            | **对应的代码部分**                                           |
| ------------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| **数据：Lena 灰度图**                      | **已实现**                  | 代码加载 `lena/lena512.bmp` 文件。                           |
| **要求：调用函数实现 Canny 边缘检测**      | **已实现**                  | 使用 OpenCV 库中的 `cv2.Canny(img_gray, T_low, T_high)` 函数直接调用 Canny 算法。 |
| **要求：比较不同参数下检测结果**           | **已实现**                  | 定义了三组不同的双阈值参数 (`A_Loose`, `B_Standard`, `C_Strict`)，并循环运行和保存了结果。 |
| **要求：学习 Canny 边缘检测源代码 (原理)** | **已实现 (已在指导中给出)** | 原理（高斯滤波、计算梯度、NMS、双阈值跟踪）已在指导中详细解释，需要体现在最终的**作业报告**中。 |
| **结果输出**                               | **已实现**                  | 代码生成并保存了三张 Canny 结果图，同时生成一张包含原始图和三种结果的**汇总对比图** (`Canny_Comparison_Summary.png`)。 |

代码如下：


```python
# ==============================================================================
# 计算机视觉第一次作业 - 任务二：Canny 边缘检测子实现
#
# 功能: 对 Lena 灰度图执行 Canny 边缘检测，并对比不同双阈值参数的影响。
# 环境: Python 3.x, numpy, opencv-python (cv2), matplotlib
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 参数设置 ---
# Canny 阈值组 (T_low, T_high)
CANNY_PARAMS = {
    'A_Loose': (30, 90),    # 宽松组：细节和噪声更多
    'B_Standard': (50, 150), # 标准组：平衡效果
    'C_Strict': (100, 300)  # 严格组：只保留最强边缘
}

# 路径设置
IMAGE_BMP_PATH = 'lena/lena512.bmp'
RESULTS_DIR = 'results'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 2. 图像加载 ---
try:
    # Canny 只针对灰度图 (BMP, 单通道)
    img_gray = cv2.imread(IMAGE_BMP_PATH, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        raise FileNotFoundError("无法找到或加载灰度图片。请检查路径是否正确。")

except FileNotFoundError:
    print("致命错误: 无法加载 Lena 灰度图。请检查 'lena/lena512.bmp' 文件路径。")
    exit()

print("图像加载成功。开始执行 Canny 检测...")


# ==============================================================================
#                                  任务二：Canny 检测
# ==============================================================================

# --- 3. Canny 检测循环 ---
canny_results = {}

for name, (t_low, t_high) in CANNY_PARAMS.items():
    try:
        # cv2.Canny(image, threshold1(T_low), threshold2(T_high))
        # Canny 算法内部包含高斯滤波、梯度计算、NMS 和双阈值跟踪。
        edges = cv2.Canny(img_gray, t_low, t_high) 

        # 保存结果
        filename = f'canny_{name}_T{t_low}_{t_high}.png'
        filepath = os.path.join(RESULTS_DIR, filename)
        cv2.imwrite(filepath, edges)
        
        canny_results[name] = edges
    except Exception:
        pass


# --- 4. 结果展示与保存 (用于报告) ---
if canny_results:
    plt.figure(figsize=(15, 4))
    
    # 原始灰度图
    plt.subplot(1, 4, 1); plt.imshow(img_gray, cmap='gray'); plt.title('Original Gray Image'); plt.axis('off')
    
    i = 2
    for name, result in canny_results.items():
        t_low, t_high = CANNY_PARAMS[name]
        plt.subplot(1, 4, i); plt.imshow(result, cmap='gray'); 
        plt.title(f'{name}\n({t_low}, {t_high})'); plt.axis('off')
        i += 1
    
    plt.tight_layout()
    # 保存结果到文件，便于报告使用
    plt.savefig(os.path.join(RESULTS_DIR, 'Canny_Comparison_Summary.png'))
    plt.show() # 如果需要实时查看弹窗，保留此行

print("任务二：Canny 边缘检测已完成。")

# ==============================================================================
```




#### 二、拓展任务：数据集级检测与评估（任务三）

以 BSDS500 数据集为载体，完成边缘检测的量化评估：

1. 数据处理：读取数据集的测试图像与人工标注边缘图（Ground-Truth）；
2. 检测与对比：对测试图像执行 Canny（或其他）边缘检测，将二值检测结果与 Ground-Truth 做像素级对比；
3. 指标计算：量化评估检测性能，计算精确率（\(P=TP/(TP+FP)\)）、召回率（\(R=TP/(TP+FN)\)），可选计算 F-Measure，分析与官方结果的差距。

#### 三、提交要求

1. 代码：优先使用 Python（库函数更丰富）或 MATLAB，以自研代码为主，不打包数据；
2. 文件命名：统一为 “学号 - 姓名 - 作业 1”；
3. 报告：需包含实验内容、原理、结果及分析（结果分析是高分关键）。

整体而言，作业从 “算法实现 - 参数分析 - 量化评估” 层层递进，既考察基础图像处理算法的理解与编程能力，也注重实验结果的分析与总结，核心目标是掌握经典边缘检测算法的原理、应用与评估方法。
