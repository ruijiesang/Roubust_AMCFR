import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 模拟QPSK调制
bit_stream = np.array([0, 0, 0, 1, 1, 0, 1, 1])  # 共4个符号，每两个bit一个符号
symbols = bit_stream.reshape(-1, 2)

# QPSK星座映射规则（Gray编码）
mapping = {
    (0, 0): 1 + 1j,
    (0, 1): -1 + 1j,
    (1, 1): -1 - 1j,
    (1, 0): 1 - 1j,
}

# 生成IQ信号序列
iq_sequence = np.array([mapping[tuple(b)] for b in symbols])

# 可视化动画：星座图动画
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("In-phase (I)")
ax.set_ylabel("Quadrature (Q)")
ax.set_title("QPSK Constellation Animation")
scatter, = ax.plot([], [], 'bo', markersize=10)
text = ax.text(-1.8, 1.6, '', fontsize=12)

# 背景星座图
for key, val in mapping.items():
    ax.plot(val.real, val.imag, 'r+', markersize=12)
    ax.text(val.real + 0.1, val.imag + 0.1, f'{key}', fontsize=10)

def init():
    scatter.set_data([], [])
    text.set_text('')
    return scatter, text

def update(frame):
    point = iq_sequence[frame]
    scatter.set_data(point.real, point.imag)
    bits = symbols[frame]
    text.set_text(f'Bit: {bits[0]}{bits[1]}')
    return scatter, text

ani = animation.FuncAnimation(fig, update, frames=len(iq_sequence),
                              init_func=init, blit=True, repeat=True, interval=1000)

plt.tight_layout()
plt.show()