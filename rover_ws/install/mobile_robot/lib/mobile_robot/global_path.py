"""
global_path.py
==============
YOUR A* algorithm — moved to its own file so it can be imported by
astar_planner_node.py without any changes.

This file is IDENTICAL to the A* code you sent — zero modifications.
Just rename / copy it into:
    mobile_robot/mobile_robot/global_path.py

and add  mobile_robot.global_path  to your setup.py packages list.
"""

import numpy as np
import queue
import time
import matplotlib.pyplot as plt


def printTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        print(endTime - startTime)
        return result
    return wrapper


def euclideanDistance(s, e):
    return np.linalg.norm(np.array(s) - np.array(e))


def checkCoordinates(x, y, mapMatrix, roverDist):
    if (roverDist <= x < mapMatrix.shape[0] - roverDist and
            roverDist <= y < mapMatrix.shape[1] - roverDist and
            all([np.isfinite(z) for z in
                 mapMatrix[x - roverDist:x + roverDist + 1,
                            y - roverDist:y + roverDist + 1].reshape(-1)])):
        return True
    return False


def visualizePath(matrix, path, start, end, startPoint, endPoint):
    plt.imshow(matrix, cmap="terrain", origin="upper")
    pathX, pathY = zip(*path)
    plt.plot(pathY, pathX, color="red", linestyle="-", linewidth=1,
             marker="o", markersize=3,
             markerfacecolor="blue", markeredgecolor="blue")
    plt.plot(start[1], start[0], marker="s", markersize=10,
             markerfacecolor="green", markeredgecolor="green")
    plt.plot(end[1], end[0], marker="s", markersize=10,
             markerfacecolor="red", markeredgecolor="red")
    plt.title(f"Global Path {startPoint}-{endPoint}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.colorbar(label="Terrain Height")
    plt.grid(visible=False)
    plt.savefig(f"tests/{startPoint}-{endPoint}.png")
    plt.show()


class GlobalPath:
    def __init__(self, mapMatrix, mapMatrixLayers, roverSize,
                 distanceBetweenFields, roverMaxSlope, slopeThresholdIncrease):
        self.distanceBetweenFields = distanceBetweenFields
        self.mapMatrix             = mapMatrix
        self.mapMatrixLayers       = mapMatrixLayers
        self.roverSize    = roverSize if roverSize % 2 else roverSize + 1
        self.roverDist    = (self.roverSize - 1) // 2
        self.roverMaxSlope           = roverMaxSlope
        self.slopeThresholdIncrease  = slopeThresholdIncrease

    def calculateSlope(self, center, neighbor, searchDepth):
        centerZ    = self.mapMatrix[center[0]][center[1]]
        totalRise  = 0
        totalRun   = 0
        for dx in range(-searchDepth, searchDepth + 1):
            for dy in range(-searchDepth, searchDepth + 1):
                nx, ny = neighbor[0] + dx, neighbor[1] + dy
                if (0 <= nx < self.mapMatrix.shape[0] and
                        0 <= ny < self.mapMatrix.shape[1]):
                    nz = self.mapMatrix[nx][ny]
                    if nz != np.inf:
                        totalRise += nz - centerZ
                        totalRun  += (euclideanDistance([nx, ny], center)
                                      * self.distanceBetweenFields)
        if totalRun == 0:
            return 0
        return np.arctan(totalRise / totalRun)

    def aStarWithSlopeThreshold(self, start, end, searchDepth, slopeThreshold):
        pq        = queue.PriorityQueue()
        pq.put((0, start, [start]))
        visited   = set()
        visited.add((start[0], start[1]))
        neighbours = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]

        while not pq.empty():
            _, node, path = pq.get()
            if node == end:
                return path
            for nb in neighbours:
                x, y = node[0] + nb[0], node[1] + nb[1]
                if (checkCoordinates(x, y, self.mapMatrix, self.roverDist)
                        and (x, y) not in visited):
                    slope = self.calculateSlope(node, [x, y], searchDepth)
                    if abs(slope) < slopeThreshold:
                        cost = len(path) + euclideanDistance([x, y], end)
                        addCost = 0
                        for point in self.mapMatrixLayers[
                                x - self.roverDist:x + self.roverDist + 1,
                                y - self.roverDist:y + self.roverDist + 1
                                ].reshape(-1):
                            addCost += cost * point / pow(self.roverSize, 2)
                        cost += addCost
                        pq.put((cost, [x, y], path + [[x, y]]))
                        visited.add((x, y))

    @printTime
    def aStar(self, start, end, searchDepth=4, initialSlopeThreshold=0.05):
        path = None
        currentSlopeThreshold = initialSlopeThreshold
        while currentSlopeThreshold < self.roverMaxSlope:
            path = self.aStarWithSlopeThreshold(
                start, end, searchDepth, currentSlopeThreshold)
            if path:
                return path
            currentSlopeThreshold += self.slopeThresholdIncrease
        return path