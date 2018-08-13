""" Easter egg """
import cv2
import numpy as np


if __name__ == "__main__":
    p = [
        "....................................................",
        ".........................+..........................",
        ".......................+.+..........................",
        ".............++......++............++...............",
        "............+...+....++............++...............",
        ".++........+.....+...++.............................",
        ".++........+...+.++....+.+..........................",
        "...........+.....+.......+..........................",
        "............+...+...................................",
        ".............++.....................................",
    ]

    m = []
    for l in p:
        nl = []
        l = l.replace(".", "0")
        l = l.replace("+", "1")
        for c in l:
            nl.append(int(c))
        m.append(nl)
    orig = np.array(m, dtype=np.uint8)

    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:20+orig.shape[0], 20:20+orig.shape[1]] += orig
    no_iter = 1000
    for it in range(no_iter):
        frame = cv2.resize(m*255, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Pattern", frame)
        cv2.waitKey(1)
        print (it)
        next_m = np.zeros_like(m)
        for i in range(0, next_m.shape[0]):
            for j in range(0, next_m.shape[1]):
                c = m[i, j]
                neigh = m[i-1:i+2, j-1:j+2].sum() - c
                if c == 1:
                    if 2 <= neigh <= 3:
                        next_m[i, j] = 1
                else:
                    if neigh == 3:
                        next_m[i, j] = 1
        m = next_m

    cv2.destroyAllWindows()
