import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 参数配置
SIGMAS = [1, 3, 5, 7]  # LoG尺度因子
WINDOWS = [3, 5, 7, 9, 11]  # LoG窗口大小(奇数)
CANNY_PARAMS = {  # Canny双阈值组
    'Loose': (30, 90),
    'Standard': (50, 150),
    'Strict': (100, 300)
}

# 路径配置
IMAGE_GRAY_PATH = 'lena/lena512.bmp'
IMAGE_COLOR_PATH = 'lena/lena512color.tiff'
RESULTS_DIR = 'results'

# 创建结果目录
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# 图像加载
def load_images():
    img_color = cv2.imread(IMAGE_COLOR_PATH)
    img_gray = cv2.imread(IMAGE_GRAY_PATH, cv2.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        raise FileNotFoundError("图像文件加载失败，请检查路径")

    img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return img_gray, img_color, img_color_rgb


# 高斯滤波实现
def gaussian_filter(image, window_size, sigma):
    return cv2.GaussianBlur(image, (window_size, window_size), sigma, sigma)


# LoG边缘检测实现
def log_detection(image, window_size, sigma):
    # 高斯平滑
    smoothed = gaussian_filter(image, window_size, sigma)
    # 拉普拉斯运算
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)
    # 边缘强度计算与归一化
    edge_strength = np.abs(laplacian)
    max_val = np.max(edge_strength)
    if max_val > 0:
        edge_strength = (edge_strength / max_val) * 255
    return np.uint8(edge_strength)


# 执行LoG检测（灰度+彩色）
def run_log_detection(img_gray, img_color):
    # 灰度图LoG处理
    print("开始处理灰度图LoG检测...")
    for W in WINDOWS:
        for sigma in SIGMAS:
            try:
                log_gray = log_detection(img_gray, W, sigma)
                save_path = os.path.join(RESULTS_DIR, f'gray_W{W}_s{sigma}.png')
                cv2.imwrite(save_path, log_gray)
            except Exception as e:
                print(f"灰度图处理失败 W={W}, sigma={sigma}: {e}")

    # 彩色图分通道LoG处理
    print("开始处理彩色图LoG检测...")
    B, G, R = cv2.split(img_color)
    for W in WINDOWS:
        for sigma in SIGMAS:
            try:
                log_B = log_detection(B, W, sigma)
                log_G = log_detection(G, W, sigma)
                log_R = log_detection(R, W, sigma)
                log_color = cv2.merge([log_B, log_G, log_R])
                save_path = os.path.join(RESULTS_DIR, f'color_W{W}_s{sigma}.png')
                cv2.imwrite(save_path, log_color)
            except Exception as e:
                print(f"彩色图处理失败 W={W}, sigma={sigma}: {e}")


# 执行Canny边缘检测
def run_canny_detection(img_gray):
    print("开始执行Canny边缘检测...")
    canny_results = {}
    # Canny检测循环
    for name, (t_low, t_high) in CANNY_PARAMS.items():
        try:
            edges = cv2.Canny(img_gray, t_low, t_high)
            save_path = os.path.join(RESULTS_DIR, f'canny_{name}_T{t_low}_{t_high}.png')
            cv2.imwrite(save_path, edges)
            canny_results[name] = edges
        except Exception as e:
            print(f"Canny处理失败 {name}: {e}")
    return canny_results


# 结果可视化
def visualize_results(img_gray, img_color_rgb, canny_results):
    # LoG结果示例展示
    W_small, sigma_small = 3, 1
    W_large, sigma_large = 11, 7

    # 加载LoG示例结果
    log_gray_small = cv2.imread(os.path.join(RESULTS_DIR, f'gray_W{W_small}_s{sigma_small}.png'), cv2.IMREAD_GRAYSCALE)
    log_gray_large = cv2.imread(os.path.join(RESULTS_DIR, f'gray_W{W_large}_s{sigma_large}.png'), cv2.IMREAD_GRAYSCALE)
    log_color_small = cv2.cvtColor(cv2.imread(os.path.join(RESULTS_DIR, f'color_W{W_small}_s{sigma_small}.png')),
                                   cv2.COLOR_BGR2RGB)
    log_color_large = cv2.cvtColor(cv2.imread(os.path.join(RESULTS_DIR, f'color_W{W_large}_s{sigma_large}.png')),
                                   cv2.COLOR_BGR2RGB)

    # 绘制LoG对比图
    plt.figure(figsize=(12, 5))
    # 灰度图结果
    plt.subplot(2, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(log_gray_small, cmap='gray')
    plt.title(f'Gray LoG (W={W_small}, σ={sigma_small})')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(log_gray_large, cmap='gray')
    plt.title(f'Gray LoG (W={W_large}, σ={sigma_large})')
    plt.axis('off')

    # 彩色图结果
    plt.subplot(2, 3, 4)
    plt.imshow(img_color_rgb)
    plt.title('Original Color')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(log_color_small)
    plt.title(f'Color LoG (W={W_small}, σ={sigma_small})')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(log_color_large)
    plt.title(f'Color LoG (W={W_large}, σ={sigma_large})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'LoG_Comparison.png'))
    plt.show()

    # 绘制Canny对比图
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Gray')
    plt.axis('off')

    i = 2
    for name, result in canny_results.items():
        t_low, t_high = CANNY_PARAMS[name]
        plt.subplot(1, 4, i)
        plt.imshow(result, cmap='gray')
        plt.title(f'{name}\n(T={t_low}, {t_high})')
        plt.axis('off')
        i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'Canny_Comparison.png'))
    plt.show()


# 主函数
if __name__ == '__main__':
    try:
        # 加载图像
        img_gray, img_color, img_color_rgb = load_images()
        print("图像加载成功")

        # 执行LoG检测
        run_log_detection(img_gray, img_color)

        # 执行Canny检测
        canny_results = run_canny_detection(img_gray)

        # 结果可视化
        visualize_results(img_gray, img_color_rgb, canny_results)

        print("所有检测任务完成，结果已保存至results目录")

    except Exception as e:
        print(f"程序执行出错: {e}")