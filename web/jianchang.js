/* Local, recorded trajectories. All geometry uses normalized east/north coordinates. */
'use strict';

(() => {
  const data = window.JIANCHANG_DATA;
  const $ = id => document.getElementById(id);
  const canvas = $('mapCanvas');
  const context = canvas.getContext('2d');
  const modeNames = {cues_only: '找到三条线索即结束', street_sweep: '继续搜索可达路网', frontier: '前沿搜索', coverage: '前沿搜索与补扫'};
  let scenario = 'corridor';
  let run = null;
  let step = 0;
  let playing = false;
  let animation = 0;
  let previousTime = 0;
  let pendingSteps = 0;
  let imageLoaded = false;
  let zoom = 1;
  let pan = {x: 0, y: 0};
  let drag = null;
  let view = {width: 1, height: 1, left: 0, top: 0, mapWidth: 1, mapHeight: 1};
  const photograph = new Image();

  function setImageStatus(available, detail) {
    $('imageState').textContent = available ? '正射底图已载入' : 'OSM 路网底图';
    $('imageDetail').textContent = detail;
    document.querySelector('.dataset-state').classList.toggle('warning', !available);
    $('layerImage').disabled = !available;
    $('imageOpacity').disabled = !available;
    $('mapSubtitle').textContent = available ? '正射影像与已记录轨迹叠加' : '正射影像未就绪 · 当前显示 OSM';
  }

  function fail(message) {
    $('mapEmpty').hidden = false;
    $('mapEmpty').querySelector('p').textContent = message;
    setImageStatus(false, '案例数据未加载');
    $('sourceSummary').textContent = '本页使用同目录的案例数据与影像文件。';
  }

  function project(point) {
    return [view.left + point[0] * view.mapWidth, view.top + (1 - point[1]) * view.mapHeight];
  }

  function pathLine(points, color, width, start = 0, end = points.length - 1) {
    if (end <= start || !points.length) return;
    context.beginPath();
    const first = project(points[start]);
    context.moveTo(first[0], first[1]);
    for (let index = start + 1; index <= end; index += 1) {
      const point = project(points[index]);
      context.lineTo(point[0], point[1]);
    }
    context.strokeStyle = color;
    context.lineWidth = width;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.stroke();
  }

  function draw() {
    if (!data?.extent || !(data.extent.widthM > 0) || !(data.extent.heightM > 0) || !context) return;
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(bounds.width, 1);
    const height = Math.max(bounds.height, 1);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = '#dfe2d6';
    context.fillRect(0, 0, width, height);
    const aspect = data.extent.widthM / data.extent.heightM;
    const fittedWidth = Math.min(width, height * aspect);
    const fittedHeight = fittedWidth / aspect;
    view = {width, height, mapWidth: fittedWidth * zoom, mapHeight: fittedHeight * zoom,
      left: (width - fittedWidth * zoom) / 2 + pan.x,
      top: (height - fittedHeight * zoom) / 2 + pan.y};
    context.save();
    context.beginPath();
    context.rect(view.left, view.top, view.mapWidth, view.mapHeight);
    context.clip();
    context.fillStyle = '#e8e5dc';
    context.fillRect(view.left, view.top, view.mapWidth, view.mapHeight);
    const showPhoto = imageLoaded && $('layerImage').checked;
    if (showPhoto) {
      context.globalAlpha = Number($('imageOpacity').value) / 100;
      context.drawImage(photograph, view.left, view.top, view.mapWidth, view.mapHeight);
      context.globalAlpha = 1;
    }
    if ($('layerRoads').checked) {
      for (const feature of data.features || []) {
        if (!Array.isArray(feature.points) || feature.points.length < 2) continue;
        if (feature.kind === 'building') {
          const points = feature.points.map(project);
          context.beginPath();
          context.moveTo(...points[0]);
          points.slice(1).forEach(point => context.lineTo(...point));
          context.closePath();
          context.fillStyle = showPhoto ? '#f6d39a35' : '#a9a99a';
          context.fill();
          context.strokeStyle = showPhoto ? '#f6d39a90' : '#949887';
          context.lineWidth = .7;
          context.stroke();
        } else {
          const street = feature.kind === 'street';
          if (street && !showPhoto) pathLine(feature.points, '#fffef5', Math.max(2, 6 * Math.sqrt(zoom)));
          pathLine(feature.points, street ? (showPhoto ? '#f3d49cbb' : '#b4bda5') : '#7c8175', street ? 1.1 : 1.8);
        }
      }
    }
    if ($('layerExtent').checked && Array.isArray(data.previousExtent) && data.previousExtent.length) {
      context.setLineDash([6, 5]);
      pathLine([...data.previousExtent, data.previousExtent[0]], '#c9892d', 1.6);
      context.setLineDash([]);
    }
    if (run && $('layerRoute').checked) {
      const end = Math.min(step, run.path.length - 1);
      const firstAll = Number.isFinite(run.firstAllCuesStep) ? run.firstAllCuesStep : run.path.length;
      pathLine(run.path, '#ffffffb3', 4, 0, end);
      pathLine(run.path, '#086d70', 2, 0, Math.min(firstAll, end));
      if (end > firstAll) pathLine(run.path, '#ba4c91', 2.2, firstAll, end);
      const start = project(run.path[0]);
      context.fillStyle = '#07565a';
      context.fillRect(start[0] - 3, start[1] - 3, 6, 6);
      const current = project(run.path[end]);
      context.beginPath();
      context.arc(current[0], current[1], 12, 0, Math.PI * 2);
      context.fillStyle = '#ffffff42';
      context.fill();
      context.beginPath();
      context.arc(current[0], current[1], 5.5, 0, Math.PI * 2);
      context.fillStyle = end > firstAll ? '#b14a8e' : '#086d70';
      context.fill();
      context.strokeStyle = '#ffffff';
      context.lineWidth = 2;
      context.stroke();
    }
    if ($('layerLabels').checked) {
      context.font = '11px "Microsoft YaHei", sans-serif';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      for (const label of data.labels || []) {
        const point = project(label.point);
        const textWidth = context.measureText(label.text).width;
        context.fillStyle = '#fffdf0e6';
        context.fillRect(point[0] - textWidth / 2 - 4, point[1] - 9, textWidth + 8, 18);
        context.fillStyle = '#364d3c';
        context.fillText(label.text, point[0], point[1]);
      }
    }
    context.restore();
    const metersPerPixel = data.extent.widthM / view.mapWidth;
    const candidates = [10, 20, 50, 100, 200, 500];
    const scale = candidates.filter(value => value / metersPerPixel <= 100).pop() || 10;
    $('scaleBar').style.width = `${scale / metersPerPixel}px`;
    $('scaleBar').textContent = `${scale} 米`;
  }

  function updateStep(value) {
    if (!run) return;
    step = Math.max(0, Math.min(run.path.length - 1, Math.floor(value)));
    $('timeline').value = step;
    $('currentStep').textContent = step.toLocaleString('zh-CN');
    const firstAll = run.firstAllCuesStep;
    if (step === run.path.length - 1) {
      $('phase').textContent = run.success ? '回合已完成' : '回合结束 / 未满足完成条件';
    } else if (Number.isFinite(firstAll) && step >= firstAll) {
      $('phase').textContent = '已确认全部线索 · 继续搜索';
    } else {
      $('phase').textContent = step === 0 ? '起点 / 待播放' : '搜索与线索复核';
    }
    draw();
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(animation);
    $('playIcon').textContent = '▶';
    $('playText').textContent = '播放轨迹';
    $('playPause').setAttribute('aria-label', '播放轨迹');
  }

  function tick(time) {
    if (!playing || !run) return;
    const elapsed = Math.min((time - previousTime) / 1000, .2);
    previousTime = time;
    pendingSteps += elapsed * Number($('speed').value);
    if (pendingSteps >= 1) {
      const advance = Math.floor(pendingSteps);
      pendingSteps -= advance;
      updateStep(step + advance);
    }
    if (step >= run.path.length - 1) pause();
    else animation = requestAnimationFrame(tick);
  }

  function togglePlay() {
    if (!run) return;
    if (playing) { pause(); return; }
    if (step >= run.path.length - 1) updateStep(0);
    playing = true;
    previousTime = performance.now();
    pendingSteps = 0;
    $('playIcon').textContent = 'Ⅱ';
    $('playText').textContent = '暂停';
    $('playPause').setAttribute('aria-label', '暂停轨迹');
    animation = requestAnimationFrame(tick);
  }

  function selectRun() {
    pause();
    run = data.runs.find(candidate => candidate.scenario === scenario && candidate.mode === $('runMode').value);
    if (!run) return;
    const finalStep = run.path.length - 1;
    $('timeline').max = finalStep;
    $('totalSteps').textContent = finalStep.toLocaleString('zh-CN');
    $('stepsMetric').textContent = Number(run.steps ?? finalStep).toLocaleString('zh-CN');
    $('coverage').textContent = Number.isFinite(run.reachableCoverage) ? `${(run.reachableCoverage * 100).toFixed(1)}%` : '未记录';
    $('collisions').textContent = Number.isFinite(run.collisions) ? run.collisions : '未记录';
    $('foundMetric').textContent = Number.isFinite(run.found) ? `${run.found} 条` : '未记录';
    $('distanceMetric').textContent = Number.isFinite(run.distanceM) ? `${(run.distanceM / 1000).toFixed(2)} km` : '未记录';
    $('resultState').textContent = run.success ? '已满足本回合完成条件' : '本回合未满足完成条件';
    $('resultState').classList.toggle('incomplete', !run.success);
    $('taskDescription').textContent = run.mode === 'street_sweep'
      ? `确认线索后继续搜索，目标为可达路网探测覆盖 ${(Number.isFinite(run.requiredCoverage) ? run.requiredCoverage * 100 : 98).toFixed(0)}%。`
      : run.mode === 'cues_only' ? '确认本回合的三条合成线索后结束。可比较此时尚未探测的街区范围。'
        : '回放该条件下已保存的搜索轨迹。';
    updateStep(finalStep);
  }

  function selectScenario(next) {
    if (!data || !Array.isArray(data.runs)) return;
    const oldMode = $('runMode').value;
    scenario = next;
    document.querySelectorAll('[data-scenario]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.scenario === scenario));
    });
    const modes = [...new Set(data.runs.filter(candidate => candidate.scenario === scenario).map(candidate => candidate.mode))];
    $('runMode').replaceChildren(...modes.map(mode => {
      const option = document.createElement('option');
      option.value = mode;
      option.textContent = modeNames[mode] || mode;
      return option;
    }));
    $('runMode').value = modes.includes(oldMode) ? oldMode : modes.includes('street_sweep') ? 'street_sweep' : modes[0];
    selectRun();
  }

  function changeZoom(factor, anchor = {x: view.width / 2, y: view.height / 2}) {
    const nextZoom = Math.max(1, Math.min(8, zoom * factor));
    const scale = nextZoom / zoom;
    pan.x = anchor.x - view.width / 2 - (anchor.x - view.width / 2 - pan.x) * scale;
    pan.y = anchor.y - view.height / 2 - (anchor.y - view.height / 2 - pan.y) * scale;
    zoom = nextZoom;
    if (zoom === 1) pan = {x: 0, y: 0};
    draw();
  }

  $('playPause').addEventListener('click', togglePlay);
  $('timeline').addEventListener('input', event => { pause(); updateStep(Number(event.target.value)); });
  $('goStart').addEventListener('click', () => { pause(); updateStep(0); });
  $('goEnd').addEventListener('click', () => { pause(); updateStep(run.path.length - 1); });
  $('runMode').addEventListener('change', selectRun);
  document.querySelectorAll('[data-scenario]').forEach(button => button.addEventListener('click', () => selectScenario(button.dataset.scenario)));
  ['layerImage', 'layerRoads', 'layerRoute', 'layerLabels', 'layerExtent'].forEach(id => $(id).addEventListener('change', draw));
  $('imageOpacity').addEventListener('input', () => { $('opacityValue').textContent = `${$('imageOpacity').value}%`; draw(); });
  $('zoomIn').addEventListener('click', () => changeZoom(1.35));
  $('zoomOut').addEventListener('click', () => changeZoom(1 / 1.35));
  $('resetView').addEventListener('click', () => { zoom = 1; pan = {x: 0, y: 0}; draw(); });
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    changeZoom(Math.exp(-event.deltaY * .0015), {x: event.clientX - bounds.left, y: event.clientY - bounds.top});
  }, {passive: false});
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    drag = {x: event.clientX, y: event.clientY};
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    pan.x += event.clientX - drag.x;
    pan.y += event.clientY - drag.y;
    drag = {x: event.clientX, y: event.clientY};
    draw();
  });
  function endDrag() { drag = null; canvas.classList.remove('dragging'); }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('lostpointercapture', endDrag);
  canvas.addEventListener('keydown', event => {
    if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  window.addEventListener('resize', draw);

  if (!context) { fail('当前浏览器未启用 Canvas 绘图。请使用支持 Canvas 的浏览器。'); return; }
  if (!data?.extent || !(data.extent.widthM > 0) || !(data.extent.heightM > 0) || !Array.isArray(data.runs) || !data.runs.length) {
    fail('请检查 jianchang-assets/data.js 是否已与本页一起保存。');
    return;
  }
  data.runs = data.runs.filter(candidate => Array.isArray(candidate.path) && candidate.path.length > 1);
  if (!data.runs.length) { fail('案例文件中没有可回放的有效轨迹。'); return; }
  ['playPause', 'timeline', 'runMode', 'goStart', 'goEnd'].forEach(id => { $(id).disabled = false; });
  document.querySelectorAll('[data-scenario]').forEach(button => {
    button.disabled = !data.runs.some(candidate => candidate.scenario === button.dataset.scenario);
  });
  const extentLabel = `${Math.round(data.extent.widthM).toLocaleString('zh-CN')} × ${Math.round(data.extent.heightM).toLocaleString('zh-CN')} 米研究区`;
  $('sourceSummary').textContent = `${extentLabel}。道路来自 OpenStreetMap，轨迹来自本项目已保存的配对评估。显示坐标系 ${data.extent.crs || '由导出文件提供'}。`;
  const limits = Array.isArray(data.limitations) ? data.limitations : data.limitations ? [String(data.limitations)] : [];
  $('limitsText').textContent = limits.join(' ');
  if (data.image && data.image.available && data.image.url) {
    setImageStatus(false, '正在读取本地正射底图');
    photograph.onload = () => {
      imageLoaded = true;
      const resolution = data.image.pixelSizeM?.[0];
      const detail = Number.isFinite(resolution)
        ? `源图 ${(resolution * 100).toFixed(0)} cm/像素 · ${data.extent.crs}`
        : `本地裁片 ${data.image.width} × ${data.image.height} 像素`;
      setImageStatus(true, detail);
      draw();
    };
    photograph.onerror = () => {
      imageLoaded = false;
      setImageStatus(false, '正射文件未能读取，当前显示 OSM 路网');
      draw();
    };
    photograph.src = data.image.url;
  } else {
    setImageStatus(false, data.image?.reason || '当前导出仅包含 OSM 路网与轨迹');
  }
  selectScenario(data.runs.some(candidate => candidate.scenario === 'corridor') ? 'corridor' : data.runs[0].scenario);
})();
