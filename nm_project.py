import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def output_result(path, matrix):
    f = open(path, "w+")
    f.write(str(len(matrix)) + '\n')

    for row in matrix:
        f.write(str(row)[1:-1])
        f.write('\n')
    f.close()


def read_buildings(path_to_buildings):
    '''
    :param path_to_buildings: a path to text file that consists coordinates of the buildings
    :return: list of (x1,y1,x2,y2)
    '''
    file = open(path_to_buildings)
    buildings = []
    for line in file:
        sp = line.split(" ")

        x1, y2, x2, y1 = float(sp[0]), 1 - float(sp[1]), float(sp[2]), 1 - float(sp[3])
        buildings.append((x1, y1, x2, y2))

    return buildings


def isInsideBuildings(xx, yy, buildings):
    '''
    :param coord: (x,y) - coords to check
    :param building: (x1,y1,x2,y2) - building
    :return: Boolean
    '''

    answer = False
    for building in buildings:
        (x1, y1, x2, y2) = building

        if x1 <= xx <= x2 and y1 <= yy <= y2:
            answer = True
    return not answer


def get_cond_check_func(buildings):
    '''
    Given building construct function that verify whether point (x,y) is outside these buildings
    :param buildings:
    :return: lambda (x,y) -> ...
    '''
    return lambda x, y: isInsideBuildings(x, y, buildings)


class ConvectionDiffusion:
    def __init__(self, max_t, l1, l2, k, N, cond_func, eps):
        self.max_t = max_t
        self.l1 = l1
        self.l2 = l2
        self.k = k
        self.N = N
        self.h = 1.0 / N
        self.eps = eps
        self.cond_func = cond_func
        self.tau = 1 / (4 * k * N * N)
        self.U = np.zeros((N + 1, N + 1))
        self.coeffs = [1 - 4 * self.tau * k / (self.h * self.h),
                       self.tau * (k / (self.h * self.h) - l1 / (2 * self.h)),
                       self.tau * (k / (self.h * self.h) + l1 / (2 * self.h)),
                       self.tau * (k / (self.h * self.h) - l2 / (2 * self.h)),
                       self.tau * (k / (self.h * self.h) + l2 / (2 * self.h))]

    def check_correctness(self, x, y):
        return 0 <= x < self.N and 0 < y < self.N

    def iteration(self):
        '''
        One iteration of the simple iteration methods
        :return: error
        '''
        dx = [0, 1, -1, 0, 0]
        dy = [0, 0, 0, 1, -1]
        new_U = np.zeros((N + 1, N + 1))

        for i in range(self.N + 1):
            for j in range(self.N + 1):

                new_U[i, j] = self.U[i, j]
                if not self.cond_func(i / N, j / N):
                    continue
                else:
                    new_U[i, j] *= self.coeffs[0]

                for f in range(1, 5):
                    x = i + dx[f]
                    y = j + dy[f]
                    if self.cond_func(x / N, y / N) and self.check_correctness(x, y):
                        new_U[i, j] += self.U[x, y] * self.coeffs[f]
                    else:
                        new_U[i, j] += self.U[i, j] * self.coeffs[f]
        old_U = self.U
        self.U = new_U
        return np.max(np.abs((old_U / 100 - new_U / 100)))

    def init_matrix(self):
        self.U[:, 1] = 100

    def solve(self):
        '''
        :return: U and image
        '''
        self.init_matrix()
        for f in range(0, self.max_t):
            error = self.iteration()
            print(error)
            if error < self.eps:
                break
        fig = plt.imshow(self.U / 100)
        plt.colorbar(fig)
        plt.show()
        return self.U / 100

    def optimized_solve(self):
        vars = (self.N + 1) * (self.N + 1)
        dx = [0, 1, -1, 0, 0]
        dy = [0, 0, 0, 1, -1]
        A = sparse.lil_matrix((vars, vars))
        b = np.zeros(vars)
        frames = []
        for i in range(vars):
            y = i // (self.N + 1)
            x = i % (self.N + 1)

            A[i, i] = self.coeffs[0]
            if not self.cond_func(x / (self.N + 1), y / (self.N + 1)) or not self.check_correctness(x, y):
                continue
            if x == 0:
                A[i, i] = 1
                continue
            for j in range(1, 5):
                xx = x + dx[j]
                yy = y + dy[j]
                if xx == 0:
                    b[i] += self.coeffs[j]
                    continue
                if not self.cond_func(xx / (self.N + 1), yy / (self.N + 1)) or not self.check_correctness(xx, yy):
                    A[i, i] += self.coeffs[j]
                else:
                    A[i, yy * (self.N + 1) + xx] = self.coeffs[j]

        A = sparse.csc_matrix(A)
        x = np.zeros(vars)
        for i in range(self.N + 1):
            x[i * (self.N + 1)] = 0

        for f in range(self.max_t):
            x_new = A @ x + b
            error = np.max(np.abs((x_new - x) / np.maximum(1, x)))
            if f % 100 == 0:
                frames.append(x_new.reshape((self.N + 1, self.N + 1)))
                print(error)
            if error < self.eps:
                x = x_new
                break
            x = x_new
        answer = x.reshape((self.N + 1, self.N + 1))
        fig = plt.imshow(answer)
        plt.colorbar(fig)
        plt.show()
        return answer, frames


def animate(frames):
    fig = plt.figure()
    ims = []
    from matplotlib import animation
    i = 0
    for frame in frames:
        print(frame.shape)
        im = plt.imshow(frame, animated=True)
        ims.append([im])
        i += 1

    plt.colorbar(im)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=260)
    ani = animation.ArtistAnimation(fig, ims, interval=17, blit=True,
                                    repeat_delay=1000)
    # ani.save('diffusion.html')
    ani.save('diffusion.mp4', writer=writer)

    plt.show()


if __name__ == "__main__":
    max_t = 100000
    l_1 = 1
    l_2 = 0.0
    k = 0.5
    N = 300
    eps = 1e-6

    buildings = read_buildings("buildings.txt")
    cond_func = get_cond_check_func(buildings)
    solver = ConvectionDiffusion(max_t, l_1, l_2, k, N, cond_func, eps)
    u, frames = solver.optimized_solve()
    animate(frames)
    output_result("output.txt", u)
