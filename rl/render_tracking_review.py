"""Export matched MuJoCo replay videos for the 2026-11-08 tracking review."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import PPO

from evaluate_tracking_transfer import COMMANDS, PHASE1, SEEDS
from forest_autonomy import ForestAutonomy
from forest_tracking_residual import ForestTrackingEnv


WIDTH, HEIGHT = 512, 288
FPS = 25
COLORS = ('#9be9ce', '#ffd28e')


def export(out, name, title, detail, frames, font_file):
    title_font, label_font, body_font = [ImageFont.truetype(str(font_file), n) for n in (30, 21, 17)]
    with imageio.get_writer(str(out / f'{name}.mp4'), fps=FPS, codec='libx264', quality=8,
                            ffmpeg_params=['-movflags', '+faststart']) as writer:
        for views in frames:
            canvas = Image.new('RGB', (1056, 448), '#111d23')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 9), title, font=title_font, fill='white')
            draw.text((12, 50), detail, font=body_font, fill='#b5c4c9')
            for i, (frame, labels) in enumerate(views):
                x = 12 + i * (WIDTH + 8)
                draw.text((x, 80), ('规则基线', '规则基线 + PPO')[i], font=label_font, fill=COLORS[i])
                canvas.paste(Image.fromarray(frame), (x, 111))
                for row, label in enumerate(labels):
                    draw.text((x, 402 + 20*row), label, font=body_font, fill='white')
            last = np.asarray(canvas)
            writer.append_data(last)
        for _ in range(FPS):
            writer.append_data(last)


def tracking_frames(model):
    with ExitStack() as stack:
        states = []
        command = COMMANDS[1]
        for controller in (None, model):
            env = ForestTrackingEnv(10., 'mixed')
            stack.callback(env.close)
            obs, _ = env.reset(seed=SEEDS[0], options={'command': command})
            renderer = stack.enter_context(mujoco.Renderer(env.physics.model, height=HEIGHT, width=WIDTH))
            states.append([controller, env, renderer, obs, []])
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 2.9, 123, -23
        for step in range(250):
            views = []
            for state in states:
                controller, env, renderer, obs, errors = state
                action = np.zeros(3) if controller is None else controller.predict(obs, deterministic=True)[0]
                state[3], _, fallen, _, info = env.step(action)
                if fallen:
                    raise RuntimeError('selected tracking case fell')
                if step >= 25:
                    errors.append(info['tracking_error'])
                if step % 2 == 1:
                    camera.lookat[:] = env.physics.data.qpos[:3] + [0, 0, -.12]
                    renderer.update_scene(env.physics.data, camera=camera)
                    actual = info['actual']
                    score = np.linalg.norm(np.sqrt(np.mean(np.square(errors), axis=0)) * [1, 1, .4]) if errors else None
                    score_label = f'{score:.3f}' if score is not None else '预热中'
                    views.append((renderer.render(), (
                        f'{(step+1)*.02:4.2f} 秒   实际纵向 {actual[0]:+.2f}   横向 {actual[1]:+.2f} m/s',
                        f'实际转向 {actual[2]:+.2f} rad/s   累计跟踪误差 {score_label}')))
            if views:
                yield views


def route_frames(model, scenario, seed, stride):
    with ExitStack() as stack:
        states = []
        for controller in (None, model):
            sim = ForestAutonomy(scenario, seed, tracking_policy=controller,
                                 weighted_local=True, progress_along_route=True)
            stack.callback(sim.close)
            renderer = stack.enter_context(mujoco.Renderer(sim.env.model, height=HEIGHT, width=WIDTH))
            states.append([sim, renderer, False, None])
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = (7.6, 94, -64) if scenario == 'detour' else (3.5, 168, -62)
        for tick in range(max(s[0].env.max_steps for s in states)):
            for state in states:
                if not state[2]:
                    state[2], state[3] = state[0].step()
            finished = all(s[2] for s in states)
            if (tick+1) % stride == 0 or finished:
                views = []
                for sim, renderer, done, result in states:
                    camera.lookat[:] = [2.4, 0, .22] if scenario == 'detour' else sim.env.data.qpos[:3] + [.35, 0, -.15]
                    renderer.update_scene(sim.env.data, camera=camera)
                    scene = renderer.scene
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                        [.35, .012, 0.], [*sim.goal, .018], np.eye(3).ravel(), [.1, .55, 1., 1.])
                    scene.ngeom += 1
                    status = '到达' if result['success'] else '受阻停车' if result['blocked'] else (
                        '激光数据超时 · 撤销运动指令' if result['stale_stop'] else '跟踪路线')
                    if done and status == '跟踪路线':
                        status = '回合结束'
                    views.append((renderer.render(), (
                        f'{result["steps"]*.02:5.2f} 秒   距目标 {result["distance_m"]:.2f} m', status)))
                yield views
            if finished:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', choices=('tracking', 'detour', 'dropout'),
                        default=('tracking', 'detour', 'dropout'))
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = PPO.load(PHASE1, device='cpu')
    for clip in args.clips:
        if clip == 'tracking':
            export(args.output, '01-slope-tracking', '斜坡速度跟踪',
                   'MuJoCo 仿真   |   +10° · 同种子   |   指令 0.18 / 0.08 m/s，转向 0.16 rad/s',
                   tracking_frames(model), args.font)
        else:
            detour = clip == 'detour'
            export(args.output, '02-forest-detour' if detour else '03-lidar-dropout',
                   '林间绕障配对评测' if detour else '激光数据中断后的停车和恢复',
                   'MuJoCo 仿真   |   同一加权规划器   |   种子 10241   |   2.5 倍速' if detour else
                   'MuJoCo 仿真   |   同一加权规划器   |   种子 10240   |   5–7 秒停止刷新激光数据',
                   route_frames(model, clip, 10241 if detour else 10240, 5 if detour else 2), args.font)


if __name__ == '__main__':
    main()
