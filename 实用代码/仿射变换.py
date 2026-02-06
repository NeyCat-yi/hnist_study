import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"res\test19.jpg")
(h, w) = image.shape[:2]
image = cv2.resize(image, (w//8, h//8))
w=w//8
h=h//8
# (h, w) = image.shape[:2]

# 原始图像中的三个点
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

# 变换后对应的三个点
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# 计算仿射矩阵
M = cv2.getAffineTransform(pts1, pts2)

# 应用仿射变换
result = cv2.warpAffine(image, M, (w, h))

#显示结果
cv2.imshow("原图", image)
cv2.imshow("仿射变换后", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
