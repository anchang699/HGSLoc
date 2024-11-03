import time
import heapq

N = 3
s0 = [5,3,7,2,4,1,-1,8,6]
g = [3,5,4,8,-1,7,2,1,6]
A2_single = [[-1, 1, 2, 3, 4, 5, 6, 7, 8], [1, -1, 2, 3, 4, 5, 6, 7, 8]]


class Node:
    def __init__(self, m, fa, g, h):
        self.m = m
        self.fa = fa
        self.g = g
        self.h = h
        self.id = 0

    def __lt__(self, other):
        if (self.g + self.h) != (other.g + other.h):
            return (self.g + self.h) < (other.g + other.h)
        return self.m < other.m

        # return (self.g + self.h) < (other.g + other.h)

        # if (self.g + self.h) != (other.g + other.h):
        #     return (self.g + self.h) < (other.g + other.h)
        # return sum(abs(x) for x in self.m) < sum(abs(x) for x in other.m)


def calculate_h(id, m):
    if id == 1:
        cnt = 0
        for i in range(N * N):
            if m[i] != -1:
                if m[i] != g[i]:
                    cnt += 1
        return cnt
    elif id == 2:
        cnt = 0
        for i in range(1, N * N):
            number = i
            pos1, pos2 = 0, 0
            for j in range(N * N):
                if m[j] == number:
                    pos1 = j
                if g[j] == number:
                    pos2 = j
            line1, col1 = pos1 // N, pos1 % N
            line2, col2 = pos2 // N, pos2 % N
            cnt += abs(line1 - line2) + abs(col1 - col2)
        return cnt


start = time.time()
OPEN = []
CLOSE = []
isv = {}
dist = {}

number = 0

node = Node(s0, -1, 0, calculate_h(2, s0))
isv[tuple(node.m)] = True
dist[tuple(node.m)] = 0
heapq.heappush(OPEN, node)
while OPEN:
    top = OPEN[0]
    # OPEN.remove(top)
    heapq.heappop(OPEN)
    top.id = len(CLOSE)
    CLOSE.append(top)
    print(top.h)
    number+=1
    if top.m == g:
        print("success!")
        break
    pos = top.m.index(-1)
    line, col = pos // N, pos % N

    for delta_line, delta_col in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        new_line, new_col = line + delta_line, col + delta_col
        if 0 <= new_line < N and 0 <= new_col < N:
            new_pos = new_line * N + new_col
            child = Node(top.m[:], top.id, top.g + 1, 0)
            child.m[pos], child.m[new_pos] = child.m[new_pos], child.m[pos]
            child.h = calculate_h(2, child.m)
            if tuple(child.m) not in isv or top.g + 1 < dist[tuple(child.m)]:
                isv[tuple(child.m)] = True
                dist[tuple(child.m)] = top.g + 1
                # OPEN.append(child)
                heapq.heappush(OPEN, child)

end = time.time()

temp = []
trace = CLOSE[-1]
temp.append(trace)
while True:
    if trace.fa == -1:
        break
    trace = CLOSE[trace.fa]
    temp.append(trace)
cnt = 0
si = ""
A2_single.clear()
for k in range(len(temp) - 1, -1, -1):
    si += "步数：" + str(cnt) + "\r\n"
    cnt += 1
    A2_single.append(temp[k].m)
    for i in range(N):
        for j in range(N):
            si += str(temp[k].m[i * N + j]) + " "
        si += "\r\n"

A2_time = end - start
A2_open_num = len(OPEN)
A2_close_num = len(CLOSE)
A2_steps = si
A2_step_num = cnt - 1

print("time:", A2_time)
print("open num:", A2_open_num)
print("close num:", A2_close_num)
print("step:\n", A2_steps)
print("step num:", A2_step_num)

print(number)