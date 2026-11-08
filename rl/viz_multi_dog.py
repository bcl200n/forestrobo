# -*- coding: utf-8 -*-
"""多狗协同搜索实时可视化 (10只狗, 带动画)"""
import math
import os
import sys
import tkinter as tk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_dog_search import MultiDogCoordinator

CELL = 28  # 每格像素
SPEED_MS = 60  # 每步间隔


class MultiDogViz:
    def __init__(self, n_dogs=10, world_size=20, seed=42):
        self.sim = MultiDogCoordinator(world_size=world_size, seed=seed,
                                       n_dogs=n_dogs)
        self.n = world_size
        self.cell = CELL
        self.w = self.n * self.cell
        self.h = self.n * self.cell + 80

        self.root = tk.Tk()
        self.root.title(f'多狗协同搜索 — {n_dogs}只狗')
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h,
                                bg='#1a1a2e')
        self.canvas.pack()

        self.info = tk.Label(self.root, text='', font=('Consolas', 11),
                             fg='#e0e0e0', bg='#1a1a2e')
        self.info.pack(fill=tk.X)

        self.running = True
        self.trail = {}  # dog_id -> [(x,y),...]

        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.step()
        self.root.mainloop()

    def close(self):
        self.running = False
        self.root.destroy()

    def step(self):
        if not self.running:
            return
        done = self.sim.step()
        self.draw()
        if done:
            self.info.config(
                text=f'✓ 全部找到! 步数={self.sim.step_count} '
                     f'存活={sum(d.alive for d in self.sim.dogs)}')
            return
        self.root.after(SPEED_MS, self.step)

    def draw(self):
        c = self.canvas
        c.delete('all')

        # 网格背景
        for x in range(self.n):
            for y in range(self.n):
                x1, y1 = x * self.cell, y * self.cell
                x2, y2 = x1 + self.cell, y1 + self.cell
                if self.sim.blocked[x][y]:
                    c.create_rectangle(x1, y1, x2, y2, fill='#3d3d5c',
                                      outline='#2a2a44')
                else:
                    # 已探索区域
                    explored = any((x, y) in d.known for d in self.sim.dogs)
                    if explored:
                        c.create_rectangle(x1, y1, x2, y2, fill='#263238',
                                          outline='#1e272e')
                    else:
                        c.create_rectangle(x1, y1, x2, y2, fill='#1a1a2e',
                                          outline='#151525')

        # 目标(幸存者)
        for i, target in enumerate(self.sim.targets):
            tx, ty = target
            cx = (tx + 0.5) * self.cell
            cy = (ty + 0.5) * self.cell
            if i in self.sim.found:
                c.create_oval(cx-8, cy-8, cx+8, cy+8, fill='#4caf50',
                              outline='#66bb6a', width=2)
                c.create_text(cx, cy, text='✓', fill='white',
                              font=('Arial', 9, 'bold'))
            else:
                c.create_oval(cx-8, cy-8, cx+8, cy+8, fill='#ff5252',
                              outline='#ff7043', width=2, dash=(3, 2))

        # 狗
        dog_colors = ['#4fc3f7', '#ffb74d', '#81c784', '#e57373', '#ba68c8',
                      '#4dd0e1', '#f06292', '#aed581', '#ff8a65', '#9575cd',
                      '#4db6ac', '#fff176', '#ff8a80', '#7986cb', '#a1887f',
                      '#90a4ae']

        for dog in self.sim.dogs:
            if not dog.alive:
                continue
            cx = (dog.pos[0] + 0.5) * self.cell
            cy = (dog.pos[1] + 0.5) * self.cell
            color = dog_colors[dog.id % len(dog_colors)]

            # 目标线
            if dog.assigned_target:
                tx, ty = dog.assigned_target
                tpx = (tx + 0.5) * self.cell
                tpy = (ty + 0.5) * self.cell
                c.create_line(cx, cy, tpx, tpy, fill=color, dash=(2, 3),
                              stipple='gray50')

            # 狗本体 (三角形指示朝向)
            size = 7
            c.create_polygon(
                cx - size, cy + size, cx + size, cy + size, cx, cy - size,
                fill=color, outline='white', width=1)
            c.create_text(cx, cy + size + 8, text=str(dog.id),
                          fill=color, font=('Consolas', 8))

        # 信息栏
        found = len(self.sim.found)
        total = len(self.sim.targets)
        alive = sum(d.alive for d in self.sim.dogs)
        coverage = min(sum(len(d.known) for d in self.sim.dogs) /
                       (self.n * self.n) * 100, 100)
        self.info.config(
            text=f'步数={self.sim.step_count}  '
                 f'找到={found}/{total}  存活={alive}/{self.sim.n_dogs}  '
                 f'覆盖={coverage:.0f}%  '
                 f'任务={sum(1 for d in self.sim.dogs if d.assigned_target)}')


if __name__ == '__main__':
    MultiDogViz(n_dogs=10, seed=42)
