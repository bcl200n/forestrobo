// 雷波林业局机器狗巡护部署方案报告生成器
// 结构: 封面(R1/FG-1) + 目录(罗马页码) + 正文(阿拉伯页码, 10章+3附录)
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableLayoutType, SectionType, TableOfContents,
} = require("docx");
const fs = require("fs");
const path = require("path");

const ASSETS = path.join(__dirname, "assets");
const PLAN = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "rl", "deployment_plan.json"), "utf-8"));

// ---------- FG-1 Forest Mint 配色 ----------
const P = {
  bg: "0C1F1A", accent: "3DDBB5",
  titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078",
  heading: "1F4D40", body: "000000", secondary: "5A6B66",
  tHeaderBg: "2A7A65", tHeaderText: "FFFFFF", tInner: "C5D8D0", tSurface: "EDF5F2",
};

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

// ---------- 封面工具 (design-system) ----------
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."\uff0c\u3002\u3001\uff1b\uff1a\uff01\uff1f", ..."\u7684\u4e0e\u548c\u53ca\u4e4b\u5728\u4e8e\u4e3a", ..."-_\u2014\u2013\u00b7/", ..." \t"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) {
      breakAt = charsPerLine;
      const prevChar = remaining[breakAt - 1], nextChar = remaining[breakAt];
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar) &&
          /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) breakAt -= 1;
    }
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / (pt * 20));
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, charsPerLine(minPt));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
          metaLineCount = 0, fixedHeight = 800, pageHeight = 16838 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const safeRemaining = Math.max(usableHeight - contentHeight, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  return { topSpacing, midSpacing: Math.max(safeRemaining - topSpacing - bottomSpacing, 0), bottomSpacing };
}

function buildCoverR1(config) {
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: P.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: P.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: P.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: P.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: P.footerColor, font: { ascii: "Arial", eastAsia: "Microsoft YaHei" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders,
        verticalAlign: "top",
        children,
      })],
    })],
  })];
}

// ---------- 正文构件 ----------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    keepNext: true, // 标题不孤悬页底
    spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: P.heading, font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    keepNext: true,
    spacing: { before: 240, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 28, color: P.heading, font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: opts.after || 80 },
    children: [new TextRun({ text, size: 24, color: P.body, font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function bodyRuns(runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 80 },
    children: runs.map(r => new TextRun({ text: r.t, bold: !!r.b, size: 24, color: P.body, font: { eastAsia: "SimSun", ascii: "Times New Roman" } })),
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 200, line: 312 },
    children: [new TextRun({ text, size: 21, color: P.secondary, font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function tableTitle(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80, line: 312 },
    keepNext: true,
    children: [new TextRun({ text, bold: true, size: 21, color: P.body, font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}

function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function figure(file, displayWidth) {
  const buf = fs.readFileSync(path.join(ASSETS, file));
  const { w, h } = pngSize(buf);
  const dw = displayWidth, dh = Math.round(displayWidth * h / w);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: true, // 图与图注同页
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({ data: buf, transformation: { width: dw, height: dh }, type: "png" })],
  });
}

// 通用数据表 (Horizontal-Only 风格, FG-1 色板)
function dataTable(headers, rows, colWidths, fontSize = 18) {
  const mk = (text, isHeader, i) => new TableCell({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { line: 276 },
      children: [new TextRun({ text: String(text), bold: isHeader, size: fontSize,
        color: isHeader ? P.tHeaderText : P.body,
        font: { eastAsia: isHeader ? "SimHei" : "SimSun", ascii: "Times New Roman" } })],
    })],
    shading: isHeader ? { type: ShadingType.CLEAR, fill: P.tHeaderBg } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    width: colWidths ? { size: colWidths[i], type: WidthType.PERCENTAGE } : undefined,
    verticalAlign: "center",
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: P.tHeaderBg },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: P.tHeaderBg },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: P.tInner },
      insideVertical: NB,
    },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((t, i) => mk(t, true, i)) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((t, i) => mk(t, false, i)) })),
    ],
  });
}

function pageNumFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })],
    })],
  });
}
function docHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: P.tInner, space: 4 } },
      children: [new TextRun({ text: "雷波林业局机器狗巡护部署方案", size: 18, color: "808080",
        font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
    })],
  });
}

// ---------- 数据准备 ----------
const farms = ["211", "212", "213", "214"];
const agg = {};
for (const f of farms) {
  const fz = PLAN.zones.filter(z => z.farm === f);
  const fm = PLAN.meta.farms[f];
  agg[f] = {
    zones: fz, dogs: fz.length,
    mu: fz.reduce((a, z) => a + z.mu, 0),
    comps: fz.reduce((a, z) => a + z.n, 0),
    km: fz.reduce((a, z) => a + z.km, 0),
    h: fz.reduce((a, z) => a + z.hours, 0),
    seg: fz.reduce((a, z) => a + z.sorties, 0),
    sw: fz.reduce((a, z) => a + z.swaps, 0),
    wh: fz.reduce((a, z) => a + z.wh, 0) / 1000,
    slope: fm.slope,
  };
}
const T = {
  dogs: farms.reduce((a, f) => a + agg[f].dogs, 0),
  mu: farms.reduce((a, f) => a + agg[f].mu, 0),
  comps: farms.reduce((a, f) => a + agg[f].comps, 0),
  km: farms.reduce((a, f) => a + agg[f].km, 0),
  h: farms.reduce((a, f) => a + agg[f].h, 0),
  seg: farms.reduce((a, f) => a + agg[f].seg, 0),
  sw: farms.reduce((a, f) => a + agg[f].sw, 0),
  wh: farms.reduce((a, f) => a + agg[f].wh, 0),
};

// ---------- 正文 ----------
const bodyChildren = [];

// 一、摘要
bodyChildren.push(h1("一、执行摘要"));
bodyChildren.push(body("本报告面向雷波林业局 (局编号 9120) 下辖 211、212、213、214 四个林场的日常巡护与林下作业需求，基于 271 个林班的真实边界数据与约 30 米分辨率的真实数字高程模型 (DEM)，完成林下作业四足机器狗的机队部署规划，给出机队规模、责任区划分、补给站选址、巡护路线以及电池保障体系的完整方案。"));
bodyChildren.push(bodyRuns([
  { t: "规划结论为：全局配置 " },
  { t: "31 只机器狗、31 个责任区与 31 个补给站", b: true },
  { t: "，覆盖 3.49 万亩巡护面积；引入真实地形后，巡护总耗时由平地模型的 147 小时增至 339 小时 (增幅 131%)，地形因素使机队规模由 28 只调整为 31 只；按双包 1500 Wh 电池与每日 2 段出勤计算，全局每巡护循环需换电 71 次、总能耗 114.7 千瓦时，巡护周期不超过 3 天。" },
]));
bodyChildren.push(body("主要建议：优先获取森林二类调查小班属性 (林相) 与林区道路网数据以替换当前的直线距离假设；在机型定型后实测功耗与坡道速度曲线，校准能耗模型；先行选取 1 至 2 个责任区开展试点运行。"));

// 二、背景
bodyChildren.push(h1("二、项目背景与规划目标"));
bodyChildren.push(h2("2.1 项目背景"));
bodyChildren.push(body("林下作业四足机器狗项目旨在面向西南山地林区提供巡护、森林碳汇调查、修枝疏伐等作业能力。单机层面已完成轮足与足式两种构型的运动控制与强化学习训练 (仿真环境平滑地形存活率 80%、山地 65%)，以及修枝剪与圆锯复合工具头等执行机构设计。在单机能力具备的基础上，机队层面的部署问题成为工程化的核心环节：需要多少台机器狗、如何划分责任区域、站点设在哪里、按什么路线巡护、电池如何保障。"));
bodyChildren.push(h2("2.2 规划目标与范围"));
bodyChildren.push(body("本报告解决机队部署规划问题，具体包括四项任务：一是确定四个林场各自所需的机器狗数量与责任区划分；二是为每个责任区选定补给站建议位置；三是生成各责任区的闭合巡护路线并优化访问顺序；四是建立电池续航、换电时间与出勤节奏的保障模型。"));
bodyChildren.push(body("本报告属于规划层面，不涉及单机感知导航与运动控制的实现细节 (该部分由项目强化学习与自主导航模块另行覆盖)，亦不包含机队采购成本测算。"));

// 三、数据
bodyChildren.push(h1("三、数据基础与数据源"));
bodyChildren.push(body("规划采用三类真实数据源，全部为实际测绘或公开地理数据，未使用虚构边界。"));
bodyChildren.push(tableTitle("表 3-1 数据源清单"));
bodyChildren.push(dataTable(
  ["数据源", "内容", "规模与精度"],
  [
    ["forest_compartments.geojson", "271 个林班边界多边形及属性 (局、场、作业区、林班号、面积、周长)", "WGS84 坐标, 面积 56.3 至 3721.6 亩"],
    ["Mapzen terrarium DEM 瓦片", "覆盖 103.30 至 103.72 E、28.28 至 28.66 N 的数字高程模型", "36 张 z12 瓦片, 约 30 米/像素, 高程 286 至 3827 米"],
    ["林下经济与直服用地 KMZ", "石斛、黄莲等林下经济地块边界 (7 处)", "WGS84 坐标"],
  ],
  [28, 44, 28], 18));
bodyChildren.push(body("各林场管辖范围与地形概况如表 3-2 所示。四个林场平均坡度均在 21 度以上，最大坡度接近 34 度，海拔高差普遍超过 1000 米，属于典型的高山峡谷林区，地形对巡护效率的影响不可忽略。"));
bodyChildren.push(tableTitle("表 3-2 四林场基础概况 (DEM 实测)"));
bodyChildren.push(dataTable(
  ["林场", "林班数", "面积 (万亩)", "平均坡度", "最大坡度", "海拔范围 (米)"],
  [
    ["211", "27", "0.37", "24.8 度", "31.6 度", "995 - 2340"],
    ["212", "105", "1.01", "23.0 度", "33.6 度", "1423 - 2670"],
    ["213", "62", "0.65", "22.8 度", "29.5 度", "1371 - 2609"],
    ["214", "77", "1.46", "21.4 度", "33.0 度", "1171 - 2271"],
  ],
  [12, 14, 18, 18, 18, 20], 18));

// 四、方法
bodyChildren.push(h1("四、部署规划方法"));
bodyChildren.push(h2("4.1 责任区划分"));
bodyChildren.push(body("责任区划分采用面积加权 K 均值聚类：以各林班质心为聚类对象、林班面积为权重，在各林场内部将林班聚为若干责任区，每个责任区配置 1 只机器狗。权重使站点与路线自动向大面积林班倾斜。聚类结果再经两条约束规则修正：其一，巡护路线几何长度超过 20 公里或单循环出勤段数超过 6 段的责任区自动二分；其二，少于 3 个林班的碎片责任区合并至同林场内地理最近的相邻区。"));
bodyChildren.push(h2("4.2 补给站选址"));
bodyChildren.push(body("补给站位置取责任区内林班面积的加权质心，保证站点位于其服务林班的负荷中心。站址为规划建议值，实施时应结合林区道路可达性、电源条件与通信覆盖进行实地校准，校准幅度一般不超过数百米，对本报告的机队规模结论无实质影响。"));
bodyChildren.push(h2("4.3 巡护路线优化"));
bodyChildren.push(body("各责任区巡护路线建模为以补给站为起终点的闭合巡回售货商问题 (TSP)，采用三步求解：最近邻贪心构造初始解，2-opt 边交换消除交叉边，Or-opt 段搬迁 (长度 1 至 3) 进一步改善。基准实验表明，2-opt 与 Or-opt 相对最近邻初始解缩短路线 6.9% (322.4 公里降至 300.3 公里)。"));
bodyChildren.push(body("路线边权采用地形加权代价：边代价等于直线距离乘以 (1 + (坡度 / 22 度) 的平方)，坡度沿边每 250 米采样一次，使翻越山脊的连接自动变贵。对比实验显示，在 31 个责任区中有 23 个因地形代价改变了林班访问顺序，地形代价平均节省 3.4%。这一结果表明：地形对规划的主要影响体现在耗时与能耗 (见第五章、第六章)，而非访问顺序本身。"));

// 五、地形
bodyChildren.push(h1("五、地形影响分析"));
bodyChildren.push(body("图 5-1 为全局部署总图，山体阴影反映了真实地形起伏，四个林场以不同颜色区分，黑色三角为补给站，细线为各责任区闭合巡护路线。"));
bodyChildren.push(figure("map_overview.png", 560));
bodyChildren.push(caption("图 5-1 雷波林业局四林场机器狗部署总图 (真实 DEM 山体阴影 + 271 林班边界 + 巡护路线 + 补给站)"));
bodyChildren.push(body("图 5-2 给出 271 个林班平均坡度的分布。全部林班坡度集中于 15 至 30 度区间，211 场整体最陡 (平均 24.8 度)，214 场因含大面积缓坡台地而相对平缓 (平均 21.4 度)。"));
bodyChildren.push(figure("slope_hist.png", 560));
bodyChildren.push(caption("图 5-2 各林场林班坡度分布 (DEM 30 米采样)"));
bodyChildren.push(body("速度模型采用 v = 0.6 × exp(−坡度 / 32 度) (米/秒)，其中 0.6 米/秒为林下平地巡航速度。坡度按横坡处理，即假定巡护主要沿等高线进行，30 度坡地仍保留约 39% 的平地速度。该模型为经验假设，建议机型定型后以实测坡道速度曲线替换。"));
bodyChildren.push(bodyRuns([
  { t: "引入地形变速后，全局巡护总耗时由平地匀速模型的 147 小时增至 " },
  { t: "339 小时，增幅 131%", b: true },
  { t: "。若沿用平地假设，各责任区的巡护周期将被系统性低估至一半以下，这是山地林区机队规划中必须显式建模的因素。" },
]));

// 六、电池
bodyChildren.push(h1("六、电池续航与换电保障"));
bodyChildren.push(h2("6.1 能耗模型"));
bodyChildren.push(body("单边能耗由基础功耗与地形做功两部分构成：E = P × t + m × g × Δh上升 / η − m × g × Δh下降 × 回收率。其中爬坡位能项的高差 Δh 取自 DEM 沿线采样，下坡按 30% 回收。模型参数见表 6-1。"));
bodyChildren.push(tableTitle("表 6-1 电池与能耗模型参数 (50 公斤级林下作业四足)"));
bodyChildren.push(dataTable(
  ["参数", "取值", "说明"],
  [
    ["整机质量", "50 kg", "整机加作业载荷"],
    ["电池组", "1500 Wh 双包", "2 × 750 Wh 热插拔轮换"],
    ["可用放电深度", "80%", "留 20% 保护裕量, 可用 1200 Wh"],
    ["步态基础功率", "180 W", "0.6 米/秒林下慢速步态"],
    ["爬坡机电效率", "0.75", "位能做功折算"],
    ["下坡能量回收", "30%", "制动回馈"],
    ["换电耗时", "15 分钟/次", "换电加自检"],
    ["充电时长", "1.5 小时/组", "充电柜双充"],
    ["出勤节奏", "2 段/日", "上午与下午各一段"],
  ],
  [26, 24, 50], 18));
bodyChildren.push(h2("6.2 出勤切段策略"));
bodyChildren.push(body("机器狗沿闭合巡回路线行进时，实时校核剩余能量：当剩余能量不足以覆盖下一条边加返站直达路程时，即就地折返补给站换电，新出勤段自站点直奔折返点继续巡护。该策略保证任意时刻机器狗不会因亏电滞留站外。规划结果为全局每巡护循环换电 71 次，总能耗 114.7 千瓦时。"));
bodyChildren.push(body("参数校准过程中得到一个重要教训：若按 800 Wh 单包与 250 W 功耗配置计算，可用能量仅支撑约 2.6 小时行走，各责任区被切分为 4 至 8 个出勤段，往返站点的空驶时间反超巡护本身，机队规模膨胀至 60 只且伴随大量无效往返。该配置被物理约束直接否决，说明电池容量与整机功耗是部署可行性的先决条件，须在选型阶段锁定。"));
bodyChildren.push(figure("battery_bars.png", 560));
bodyChildren.push(caption("图 6-1 各林场电池保障负荷 (出勤段、换电次数与循环能耗)"));
bodyChildren.push(body("规划中同时标记了 8 条超程边，即满电状态下无法连同返程一并走完的单边，全部位于 214 场的陡峭跨沟地段。实施时应对这些边增设途中补电点，或调整对应责任区边界。"));

// 七、结果
bodyChildren.push(h1("七、机队配置与统计结果"));
bodyChildren.push(body("表 7-1 给出按林场的机队配置统计。机队规模由责任区数决定，而责任区数取地理聚类、路线长度上限 (20 公里) 与电池约束 (单循环不超过 6 个出勤段) 三者中最紧的约束。212 与 214 两场因林班数量多、面积大，其责任区数正是由电池约束顶至 10 个。"));
bodyChildren.push(tableTitle("表 7-1 各林场机队配置统计"));
bodyChildren.push(dataTable(
  ["林场", "机器狗", "林班", "面积 (万亩)", "平均坡度", "路线 (公里)", "地形耗时 (小时)", "出勤段", "换电", "能耗 (千瓦时)"],
  [
    ...farms.map(f => [
      f, String(agg[f].dogs), String(agg[f].comps), (agg[f].mu / 10000).toFixed(2),
      agg[f].slope.toFixed(1) + " 度", agg[f].km.toFixed(1), agg[f].h.toFixed(1),
      String(agg[f].seg), String(agg[f].sw), agg[f].wh.toFixed(1),
    ]),
    ["合计", String(T.dogs), String(T.comps), (T.mu / 10000).toFixed(2), "-", T.km.toFixed(1), T.h.toFixed(1), String(T.seg), String(T.sw), T.wh.toFixed(1)],
  ],
  [8, 10, 8, 13, 12, 12, 13, 9, 7, 8], 16));
bodyChildren.push(body("图 7-1 直观对比了各林场的机队数量、林班数量与巡护面积。214 场面积最大 (1.46 万亩，占全局 42%) 且能耗最高；212 场林班数最多 (105 个) 但单班面积小；211 场面积最小却坡度最陡，单狗负荷并不轻。"));
bodyChildren.push(figure("fleet_bars.png", 560));
bodyChildren.push(caption("图 7-1 各林场机队配置与巡护面积对比"));
bodyChildren.push(body("配套保障需求按机队规模推算：全局需 1500 Wh 电池组 62 组 (每狗 2 组轮换，一组作业一组站充)、补给站 31 个 (每责任区 1 个，含充电柜与工具头库)；按巡护周期不超过 3 天计，全局日均充电量约 38 千瓦时，其中 211 场约 4.4、212 场约 11.3、213 场约 7.4、214 场约 15.2 千瓦时。全部 31 个责任区的明细参数见附录 A。"));

// 八、节奏
bodyChildren.push(h1("八、实施节奏与任务体系"));
bodyChildren.push(body("机队部署建议按四个阶段渐进推进。第一阶段为踏查建图：全部责任区各巡护一遍，建立林区高精度地图 (地形、障碍、可通行走廊)，为后续作业提供底图；第二阶段为碳汇调查：按样地方案开展胸径、树高抽样测量，机器狗携带 DBH 卡尺与取样钻执行；第三阶段为修枝疏伐作业：搭载修枝剪与圆锯复合工具头按小班推进，补给站承担工具头更换；第四阶段为灾害响应：地震或火情场景下机队切换为多机协同模式，与项目已完成的多机搜索、地震韧性、清障协作调度模块衔接。"));
bodyChildren.push(body("日常巡护按每责任区 1 只狗定编，巡护周期不超过 3 天 (每日 2 个出勤段)。防火期如需提升至每日全巡，可按 1.5 至 2 倍系数增配机动组，或对高风险林班单独加密巡护频次。"));

// 九、风险
bodyChildren.push(h1("九、风险与限制"));
bodyChildren.push(tableTitle("表 9-1 主要风险与限制"));
bodyChildren.push(dataTable(
  ["类别", "描述", "影响与对策"],
  [
    ["站位未校准", "补给站取面积加权质心, 未结合道路与电源", "实施前实地校准, 偏移数百米内不影响结论"],
    ["直线边假设", "路线边按直线计算, 无林区道路网数据", "引入路网后用 A-star 路网距离替换, 算法管线已就绪"],
    ["林相未接入", "无树种、郁闭度、灌木密度属性", "速度模型暂不区分林相, 获取二类调查数据后按小班生效"],
    ["超程边", "214 场 8 条单边满电无法走完含返程", "增设途中补电点或微调责任区边界"],
    ["速度与功耗为经验值", "0.6 米/秒与 180 W 为文献与仿真推定", "机型定型后实测标定, 参数集中于部署脚本顶部可一键重算"],
    ["DEM 分辨率", "30 米格网, 沟坎细节被平滑", "重点走廊可补采 10 米级 DEM 局部替换"],
  ],
  [16, 42, 42], 16));

// 十、结论
bodyChildren.push(h1("十、结论与后续建议"));
bodyChildren.push(body("本报告基于真实林班边界与真实地形数据，建立了包含责任区聚类、地形加权路线优化与电池能耗保障的完整部署规划链路，结论为：211 场 4 只、212 场 10 只、213 场 7 只、214 场 10 只，全局 31 只机器狗、31 个补给站、62 组电池，巡护周期不超过 3 天。地形使巡护耗时增加 131%，是机队规模的第一决定因素；电池容量与功耗是部署可行性的先决条件。"));
bodyChildren.push(body("后续工作建议按优先级推进：一是由林业局提供森林二类调查小班属性表与林区道路网数据，替换直线边假设并使林相系数按小班生效；二是机型定型后开展坡道速度与功耗实测，校准速度与能耗模型；三是选取 211 场一个责任区 (建议 4 至 7 个林班的区) 开展试点，以实测数据回归修正全局参数；四是结合防火期任务剖面，评估机动组增配方案。"));

// 附录 A
bodyChildren.push(h1("附录 A 责任区明细表"));
bodyChildren.push(body("下表列出全部 31 个责任区的主要参数。海拔列为路线沿线最低与最高值。", { after: 40 }));
bodyChildren.push(tableTitle("表 A-1 31 个责任区明细"));
bodyChildren.push(dataTable(
  ["责任区", "林班", "面积 (亩)", "坡度", "路线 (公里)", "耗时 (小时)", "出勤段", "换电", "周期 (天)", "海拔 (米)"],
  PLAN.zones.map(z => [
    z.id, String(z.n), z.mu.toFixed(0), z.member_slope.toFixed(1) + " 度",
    z.km.toFixed(1), z.hours.toFixed(1), String(z.sorties), String(z.swaps),
    String(z.cycle), z.elev.replace("/", " - "),
  ]),
  [11, 7, 11, 10, 11, 11, 8, 7, 10, 14], 16));

// 附录 B
bodyChildren.push(h1("附录 B 规划参数汇总"));
bodyChildren.push(tableTitle("表 B-1 部署规划模型参数"));
bodyChildren.push(dataTable(
  ["模块", "参数", "取值"],
  [
    ["责任区", "初始聚类数 (211/212/213/214)", "3 / 8 / 6 / 8"],
    ["责任区", "路线几何上限", "20 公里"],
    ["责任区", "单循环出勤段上限 / 最小拆分区", "6 段 / 5 个林班"],
    ["地形", "边采样间距 / 边代价系数", "250 米 / 22 度平方项"],
    ["地形", "速度模型", "0.6 × exp(−坡度/32度) 米/秒"],
    ["路线", "TSP 求解", "最近邻 + 2-opt + Or-opt"],
    ["电池", "可用能量 / 基础功率", "1200 Wh / 180 W"],
    ["电池", "换电 / 充电 / 出勤节奏", "15 分钟 / 1.5 小时 / 2 段每日"],
  ],
  [14, 44, 42], 18));

// 附录 C
bodyChildren.push(h1("附录 C 交付文件清单"));
bodyChildren.push(tableTitle("表 C-1 规划交付物"));
bodyChildren.push(dataTable(
  ["文件", "内容"],
  [
    ["rl/deploy_planner.py", "部署规划主脚本 (参数集中于顶部, 可一键重算)"],
    ["rl/fetch_dem.py", "真实 DEM 下载与拼接 (Mapzen terrarium)"],
    ["rl/deployment_plan.json", "31 责任区结构化方案 (含路线坐标)"],
    ["rl/forest_deployment_demo.html", "交互式部署演示页 (卫星底图 + 巡护动画 + 换电点)"],
    ["rl/stand_params.json", "林相参数接口模板 (待二类调查数据填充)"],
    ["rl/farm_stats.py", "按场统计脚本"],
    ["data/leibo/dem/", "区域 DEM 栅格 (1536 × 1536, 约 30 米)"],
    ["report/", "本报告与图表生成脚本"],
  ],
  [40, 60], 18));

// ---------- 组装 ----------
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  creator: "ForestRobo 项目组",
  title: "雷波林业局林下作业机器狗巡护部署方案",
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: P.heading },
        paragraph: { spacing: { before: 360, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: P.heading },
        paragraph: { spacing: { before: 240, after: 120, line: 312 }, outlineLevel: 1 },
      },
    },
  },
  sections: [
    { // 封面
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        title: "雷波林业局林下作业机器狗巡护部署方案",
        subtitle: "基于真实林班边界与地形数据的机队配置规划报告",
        englishLabel: "PATROL DEPLOYMENT PLAN",
        metaLines: [
          "覆盖范围: 211 / 212 / 213 / 214 林场 · 271 林班 · 3.49 万亩",
          "机队规模: 31 只机器狗 · 31 个补给站 · 62 组电池",
          "数据基础: 真实边界 GeoJSON + 30 米 DEM",
          "编制: ForestRobo 项目组",
        ],
        footerLeft: "ForestRobo 项目组",
        footerRight: "2026 年 11 月",
      }),
    },
    { // 目录 (罗马页码)
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      footers: { default: pageNumFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "目  录", bold: true, size: 32, font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-1" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "注: 本目录由域代码生成。文档编辑后如需刷新页码, 请在目录上点击右键并选择更新域。",
            italics: true, size: 18, color: "888888", font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
        }),
      ],
    },
    { // 正文 (阿拉伯页码, 从 1 起)
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: docHeader() },
      footers: { default: pageNumFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "雷波林业局机器狗巡护部署方案报告.docx");
  fs.writeFileSync(out, buf);
  console.log("WRITTEN " + out + " " + buf.length + " bytes");
});
