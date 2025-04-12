import numpy as np
import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import pandas as pd

# 创建Tkinter窗口并隐藏
root = tk.Tk()
root.withdraw()

# 弹出文件选择对话框
file_path = filedialog.askopenfilename(title="选择数据文件", filetypes=[("Text files", "*.txt")])

# 确保用户选择了文件
if not file_path:
    print("没有选择文件，程序退出。")
else:
    # 读取txt文件中的数据
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]

    # 创建一个6x6的零矩阵
    D = np.zeros((6, 6))

    # 填充矩阵D的数据
    index = 0
    for i in range(6):
        for j in range(i, 6):  # 从对角线及其上方开始填充
            D[i, j] = data[index]
            index += 1

    # 将矩阵D的第一行赋值给矩阵D01
    D01 = D[0, :].copy()

    # 将D01的值四舍五入到小数点后两位，并生成预设值字符串
    D01_rounded = np.round(D01, 2)
    D01_str = ' '.join(f'{val:.4f}' for val in D01_rounded)
    D01_str_1 = ' '.join(f'{val:.4f}' for val in D01)
    # 弹出窗口让用户修改D01
    D01_str = simpledialog.askstring("编辑近似值", f"请输入近似值（用空格分隔）：\n量测值: {D01_str_1}", parent=root, initialvalue=D01_str)
    if D01_str:
        try:
            D01 = np.array([float(x) for x in D01_str.split()])
            if len(D01) != 6:
                raise ValueError("需要6个近似值。")
        except ValueError as e:
            print(f"输入错误：{e}")
            print("使用量测值。")
            D01 = D[0, :].copy()

# 生成字典Dc，键为'ij'，值为对应的D矩阵元素
    Dc = {}
    for i in range(6):
        for j in range(i, 6):  # 填充上三角矩阵，包括对角线
            key = f'{i}{j+1}'  # 键的格式为'ij'，例如'01', '12'等
            Dc[key] = D[i, j]


# D = np.array([
#     [9.8380, 25.3867, 50.9908, 87.6632, 137.1206, 202.3536],
#     [0.00000, 15.5466, 41.1536, 77.8204, 127.2736, 192.5174],
#     [0.00000, 0.00000, 25.6042, 62.2673, 111.7298, 176.9634],
#     [0.00000, 0.00000, 0.00000, 36.6594, 86.1176, 151.3543],
#     [0.00000, 0.00000, 0.00000, 0.00000, 49.4507, 114.6887],
#     [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 65.2300]
# ])



D0 = np.zeros((6, 6))
Di = np.zeros((6, 6))
D_ = np.zeros((6, 6))
Dp = np.zeros((6, 6))
Dm = np.zeros((6, 6))
VV = np.zeros((6, 6))
V = np.zeros((6, 6))
vi = np.zeros((6, 6))
l = np.zeros((6, 6))
al = bl = cl = dl = el = fl = gl = ll = 0

for i in range(6):
    D0[0, i] = D01[i]
    for j in range(1, 6):
        D0[j, i] = D01[i] - D01[j - 1]

D0 = np.where(D0 < 0, 0, D0)
l = (D0 - D) * 1000
al = np.sum(l)

for i in range(6):
    bl += l[i, 0] - l[1, i]
    cl += l[i, 1] - l[2, i]
    dl += l[i, 2] - l[3, i]
    el += l[i, 3] - l[4, i]
    fl += l[i, 4] - l[5, i]
    gl += l[i, 5]
    ll += np.sum(l[i, :] ** 2)

al = -al
a = al + bl + cl + dl + el + fl + gl
b = -ll + np.sum(l[0, :6])


B2 = np.array([
    [-1, 1, 0, 0, 0, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0],
    [-1, 0, 0, 1, 0, 0, 0],
    [-1, 0, 0, 0, 1, 0, 0],
    [-1, 0, 0, 0, 0, 1, 0],
    [-1, 0, 0, 0, 0, 0, 1],
    [-1, -1, 1, 0, 0, 0, 0],
    [-1, -1, 0, 1, 0, 0, 0],
    [-1, -1, 0, 0, 1, 0, 0],
    [-1, -1, 0, 0, 0, 1, 0],
    [-1, -1, 0, 0, 0, 0, 1],
    [-1, 0, -1, 1, 0, 0, 0],
    [-1, 0, -1, 0, 1, 0, 0],
    [-1, 0, -1, 0, 0, 1, 0],
    [-1, 0, -1, 0, 0, 0, 1],
    [-1, 0, 0, -1, 1, 0, 0],
    [-1, 0, 0, -1, 0, 1, 0],
    [-1, 0, 0, -1, 0, 0, 1],
    [-1, 0, 0, 0, -1, 1, 0],
    [-1, 0, 0, 0, -1, 0, 1],
    [-1, 0, 0, 0, 0, -1, 1]
])
P2 = np.eye(len(l))  # 单位矩阵
BT = B2.T
BPB = BT @ B2
Q = np.linalg.inv(BPB)

# Q = np.array([
#     [0.200000000, 0.057142857, 0.114285714, 0.171428571, 0.228571429, 0.285714286, 0.342857143],
#     [0.057142857, 0.302040816, 0.175510204, 0.191836735, 0.208163265, 0.224489796, 0.240816327],
#     [0.114285714, 0.175510204, 0.351020408, 0.240816330, 0.273469388, 0.306122449, 0.338775510],
#     [0.171428571, 0.191836735, 0.240816327, 0.432653061, 0.338775510, 0.387755102, 0.436734694],
#     [0.228571429, 0.208163265, 0.273469388, 0.338775510, 0.546938776, 0.469387755, 0.534693878],
#     [0.285714286, 0.224489796, 0.306122449, 0.387755102, 0.469387755, 0.693877551, 0.632653061],
#     [0.342857143, 0.240816327, 0.338775510, 0.436734694, 0.534693878, 0.632653061, 0.873469388]
# ])
print("\n近似值D0i =\n", np.round(D0, 2))
print("\n量测值Di =\n", np.round(D, 4))
print("\n差值l =\n", np.round(l, 1))
# Dc = {
#     '01': 9.8380, '02': 25.3867, '03': 50.9908, '04': 87.6632, '05': 137.1206,
#     '06': 202.3536, '12': 15.5466, '13': 41.1536, '14': 77.8204, '15': 127.2736,
#     '16': 192.5174, '23': 25.6042, '24': 62.2673, '25': 111.7298, '26': 176.9634,
#     '34': 36.6594, '35': 86.1176, '36': 151.3543, '45': 49.4507, '46': 114.6887,
#     '56': 65.2300
# }

K1 = -1/35 * (5 * (Dc['01'] + Dc['12'] + Dc['23'] + Dc['34'] + Dc['45'] + Dc['56'] - Dc['06'])
            + 3 * (Dc['02'] + Dc['13'] + Dc['24'] + Dc['35'] + Dc['46'] - Dc['05'] - Dc['16'])
            + (Dc['03'] + Dc['14'] + Dc['25'] + Dc['36'] - Dc['04'] - Dc['15'] - Dc['26']))
print("简单公式K =", np.round(K1 * 1000, 3))

# 计算 K、V01、V02、V03、V04、V05、V06
K = -(al * Q[0, 0] + bl * Q[1, 0] + cl * Q[2, 0] + dl * Q[3, 0] + el * Q[4, 0] + fl * Q[5, 0] + gl * Q[6, 0])
V01 = -(al * Q[0, 1] + bl * Q[1, 1] + cl * Q[2, 1] + dl * Q[3, 1] + el * Q[4, 1] + fl * Q[5, 1] + gl * Q[6, 1])
V02 = -(al * Q[0, 2] + bl * Q[1, 2] + cl * Q[2, 2] + dl * Q[3, 2] + el * Q[4, 2] + fl * Q[5, 2] + gl * Q[6, 2])
V03 = -(al * Q[0, 3] + bl * Q[1, 3] + cl * Q[2, 3] + dl * Q[3, 3] + el * Q[4, 3] + fl * Q[5, 3] + gl * Q[6, 3])
V04 = -(al * Q[0, 4] + bl * Q[1, 4] + cl * Q[2, 4] + dl * Q[3, 4] + el * Q[4, 4] + fl * Q[5, 4] + gl * Q[6, 4])
V05 = -(al * Q[0, 5] + bl * Q[1, 5] + cl * Q[2, 5] + dl * Q[3, 5] + el * Q[4, 5] + fl * Q[5, 5] + gl * Q[6, 5])
V06 = -(al * Q[0, 6] + bl * Q[1, 6] + cl * Q[2, 6] + dl * Q[3, 6] + el * Q[4, 6] + fl * Q[5, 6] + gl * Q[6, 6])

# 计算 vv
vv = ll + al * K + bl * V01 + cl * V02 + dl * V03 + el * V04 + fl * V05 + gl * V06
print(f"K = {np.round(K, 3)}")
print(f"V01 = {np.round(V01, 3)}")
print(f"V02 = {np.round(V02, 3)}")
print(f"V03 = {np.round(V03, 3)}")
print(f"V04 = {np.round(V04, 3)}")
print(f"V05 = {np.round(V05, 3)}")
print(f"V06 = {np.round(V06, 3)}")

V0x = [-(al * Q[0, i] + bl * Q[1, i] + cl * Q[2, i] + dl * Q[3, i] + el * Q[4, i] + fl * Q[5, i] + gl * Q[6, i]) for i in range(1, 7)]

for i in range(6):
    D01[i] += V0x[i] / 1000

for i in range(6):
    D_[0, i] = D01[i]
    for j in range(1, 6):
        D_[j, i] = D01[i] - D01[j - 1]
D_ = np.where(D_ < 0, 0, D_)

for i in range(6):
    for j in range(6):
        if D[i, j] == 0:
            Di[i, j] = 0
        else:
            Di[i, j] = D[i, j] + K / 1000

        V[i, j] = D_[i, j] - Di[i, j] - K / 1000
        if V[i, j] == -K / 1000:
            V[i, j] = 0
        V[i, j] *= 1000
        VV[i, j] = V[i, j] ** 2

vv1 = np.sum(VV)
vv2 = np.dot(V.flatten(), V.flatten())
mi = np.sqrt(vv / 14)
mk = np.sqrt(Q[0, 0]) * mi

y = V01*V01 + V02*V02 + V03*V03 + V04*V04 + V05*V05 + V06*V06
# 对 D 的非零项加上 K，得到新矩阵 Dp
Dp = np.where(D != 0, D + K/1000, D)

# Output
print("al =", np.round(al, 1))
print("bl =", np.round(bl, 1))
print("cl =", np.round(cl, 1))
print("dl =", np.round(dl, 1))
print("el =", np.round(el, 1))
print("fl =", np.round(fl, 1))
print("gl =", np.round(gl, 1))
print("ll =", np.round(ll, 1))
print("∑l =", np.round(a, 1))
print("[vv] =", np.round(vv, 1))
print("mi =", np.round(mi, 2))
print("mk =", np.round(mk, 2))
print("改正后量测值Di` =")
print(np.round(Dp, 4))
print("平差值D_ =")
print(np.round(D_, 4))
# print(np.round(D, 4))
vi = D_ - Dp
print("vi =")
print(np.round(vi*1000, 1))

# 计算矩阵各项的平方和
squared_sum = np.sum(vi**2)
xx=squared_sum*1000*1000
print("vi平方和检核[vv] =", np.round(xx, 1))


# 弹出文件选择对话框
file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])

# 确保用户选择了文件
if not file_path:
    print("没有选择文件，程序退出。")
else:
    # 创建一个ExcelWriter对象
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    # 设置工作表
    worksheet = writer.book.add_worksheet("输出结果")

    # 设置格式
    bold_format = writer.book.add_format({'bold': True})
    header_format = writer.book.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
    matrix_format = writer.book.add_format(
        {'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '0.0000'})
    number_format = writer.book.add_format(
        {'num_format': '0.0000', 'align': 'center', 'valign': 'vcenter', 'border': 1})

    # -------------------- 近似值 D0i --------------------
    worksheet.write("A1", "近似值 D0i", bold_format)
    worksheet.write("A2", "矩阵 D0i:", header_format)
    D0_output = np.round(D0, 2)
    for i in range(6):
        for j in range(i, 6):
            worksheet.write(i + 2, j + 1, D0_output[i, j], matrix_format)

    # -------------------- 量测值 Di --------------------
    worksheet.write("A8", "量测值 Di", bold_format)
    worksheet.write("A9", "矩阵 Di:", header_format)
    D_output = np.round(D, 4)
    for i in range(6):
        for j in range(i, 6):
            worksheet.write(i + 9, j + 1, D_output[i, j], matrix_format)

    # -------------------- 差值 l --------------------
    worksheet.write("A16", "差值 l", bold_format)
    worksheet.write("A17", "矩阵 l:", header_format)
    l_output = np.round(l, 1)
    for i in range(6):
        for j in range(i, 6):
            worksheet.write(i + 17, j + 1, l_output[i, j], matrix_format)

    # -------------------- 计算出的 K, V01, V02, V03, V04, V05, V06 --------------------
    worksheet.write("A24", "K 和 V 值", bold_format)
    worksheet.write("A25", "K:", header_format)
    worksheet.write("A26", np.round(K, 3), number_format)

    worksheet.write("A28", "V01:", header_format)
    worksheet.write("A29", np.round(V01, 3), number_format)
    worksheet.write("A31", "V02:", header_format)
    worksheet.write("A32", np.round(V02, 3), number_format)
    worksheet.write("A34", "V03:", header_format)
    worksheet.write("A35", np.round(V03, 3), number_format)
    worksheet.write("A37", "V04:", header_format)
    worksheet.write("A38", np.round(V04, 3), number_format)
    worksheet.write("A40", "V05:", header_format)
    worksheet.write("A41", np.round(V05, 3), number_format)
    worksheet.write("A43", "V06:", header_format)
    worksheet.write("A44", np.round(V06, 3), number_format)

    # -------------------- 改正后的量测值 Di' --------------------
    worksheet.write("A46", "改正后的量测值 Di'", bold_format)
    worksheet.write("A47", "矩阵 Di' (D_):", header_format)
    Dp_output = np.round(Dp, 4)
    for i in range(6):
        for j in range(i, 6):
            worksheet.write(i + 47, j + 1, Dp_output[i, j], matrix_format)

    # -------------------- 平差值 D_ --------------------
    worksheet.write("A54", "平差值 D_", bold_format)
    worksheet.write("A55", "矩阵 D_:", header_format)
    D_ = np.round(D_, 4)
    for i in range(6):
        for j in range(i, 6):
            worksheet.write(i + 55, j + 1, D_[i, j], matrix_format)

    # -------------------- vi --------------------
    worksheet.write("A62", "vi (平差残差)", bold_format)
    worksheet.write("A63", "矩阵 vi:", header_format)
    vi_output = np.round(vi * 1000, 1)  # 以毫米为单位
    for i in range(6):
        for j in range(6):
            worksheet.write(i + 63, j + 1, vi_output[i, j], matrix_format)

    # -------------------- 检核值 --------------------
    worksheet.write("A70", "检核值", bold_format)
    worksheet.write("A71", "vi平方和检核[vv]:", header_format)
    worksheet.write("A72", np.round(xx, 1), number_format)

    # 保存文件
    writer.close()
    print(f"文件已保存到: {file_path}")

