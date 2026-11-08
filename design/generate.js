// -*- coding: utf-8 -*-
// 林下作业机器狗专用机械臂设计方案 — DOCX 生成
// 架构: 3 节 (封面 R4/FG-1 → 前置目录 罗马页码 → 正文 阿拉伯页码)
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableOfContents, SectionType, TableLayoutType,
} = require("docx");
const fs = require("fs");
const path = require("path");

// ───────────────────────── 调色板 FG-1 Forest Mint ─────────────────────────
const P = {
  bg: "0C1F1A", accent: "3DDBB5",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "2A7A65", headerText: "FFFFFF", accentLine: "2A7A65", innerLine: "C5D8D0", surface: "EDF5F2" },
  primary: "1A2E29", body: "000000", secondary: "687078",
};
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };
const emptyPara = () => new Paragraph({ children: [] });

// ───────────────────────── 封面工具函数（design-system 标准实现）─────────────────────────
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."，。、；：！？", ..."的与和及之在于为", ..."-_—–·/", ..." \t"]);
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
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar)
        && /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) breakAt -= 1;
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
  if (!lines || lines.length > 3) { lines = splitTitleLines(title, charsPerLine(minPt)); titlePt = minPt; }
  return { titlePt, titleLines: lines };
}

// ───────────────────────── 封面 R4: Top Color Block ─────────────────────────
function buildCoverR4(config) {
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 26);
  const titleSize = titlePt * 2;
  const titleBlockHeight = titleLines.length * (titlePt * 23 + 200);
  const englishLabelH = config.englishLabel ? (9 * 23 + 500) : 0;
  const subtitleH = config.subtitle ? (12 * 23 + 200) : 0;
  const upperContentH = englishLabelH + titleBlockHeight + subtitleH;
  const UPPER_H = Math.max(7500, upperContentH + 1500 + 800);
  const DIVIDER_H = 60;
  const contentEstimate = englishLabelH + titleBlockHeight + subtitleH;
  const topSpacing = Math.max(UPPER_H - contentEstimate - 280 - 800, 400);

  const upperBlock = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: UPPER_H, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders,
        verticalAlign: "top",
        margins: { left: padL, right: padR },
        children: [
          new Paragraph({ spacing: { before: topSpacing } }),
          config.englishLabel ? new Paragraph({
            spacing: { after: 500 },
            children: [new TextRun({ text: config.englishLabel.split("").join(" "), size: 18, color: P.accent, font: { ascii: "Calibri" }, characterSpacing: 60 })],
          }) : null,
          ...titleLines.map((line, i) => new Paragraph({
            spacing: { after: i < titleLines.length - 1 ? 100 : 200, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
            children: [new TextRun({ text: line, size: titleSize, bold: true, color: P.cover.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
          })),
          config.subtitle ? new Paragraph({
            spacing: { after: 100 },
            children: [new TextRun({ text: config.subtitle, size: 24, color: P.cover.subtitleColor, font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
          }) : null,
        ].filter(Boolean),
      })],
    })],
  });

  const divider = new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: DIVIDER_H, rule: "exact" },
      children: [new TableCell({ borders: noBorders, shading: { type: ShadingType.CLEAR, fill: P.accent }, children: [emptyPara()] })],
    })],
  });

  const lowerContent = [
    new Paragraph({ spacing: { before: 800 } }),
    ...(config.metaLines || []).map(line => new Paragraph({
      indent: { left: padL }, spacing: { after: 100 },
      children: [new TextRun({ text: line, size: 28, color: P.secondary, font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    })),
    new Paragraph({ spacing: { before: 2000 } }),
    new Paragraph({
      indent: { left: padL },
      children: [
        new TextRun({ text: config.footerLeft || "", size: 22, color: "909090" }),
        new TextRun({ text: "          " }),
        new TextRun({ text: config.footerRight || "", size: 22, color: "909090" }),
      ],
    }),
  ];

  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: "FFFFFF" }, borders: noBorders,
        verticalAlign: "top",
        children: [upperBlock, divider, ...lowerContent],
      })],
    })],
  })];
}

// ───────────────────────── 正文构件 ─────────────────────────
const F_BODY = { ascii: "Times New Roman", eastAsia: "SimSun" };
const F_HEAD = { ascii: "Times New Roman", eastAsia: "SimHei" };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, keepNext: true,
    spacing: { before: 400, after: 180, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: P.primary, font: F_HEAD })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, keepNext: true,
    spacing: { before: 280, after: 130, line: 312 },
    children: [new TextRun({ text, bold: true, size: 28, color: P.primary, font: F_HEAD })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3, keepNext: true,
    spacing: { before: 220, after: 100, line: 312 },
    children: [new TextRun({ text, bold: true, size: 24, color: P.primary, font: F_HEAD })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: opts.after == null ? 60 : opts.after },
    children: [new TextRun({ text, size: 24, color: P.body, font: F_BODY })],
  });
}
// 带引导词的段落（如 （1）xxx：内容），引导词加粗
function pLead(lead, text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 60 },
    children: [
      new TextRun({ text: lead, bold: true, size: 24, color: P.body, font: F_BODY }),
      new TextRun({ text, size: 24, color: P.body, font: F_BODY }),
    ],
  });
}
function tableTitle(text) {
  return new Paragraph({
    keepNext: true, alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80, line: 312 },
    children: [new TextRun({ text, bold: true, size: 21, color: P.primary, font: F_HEAD })],
  });
}
function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 200, line: 312 },
    children: [new TextRun({ text, size: 21, color: P.secondary, font: F_HEAD })],
  });
}
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
function figure(file, caption, widthPx = 560) {
  const buf = fs.readFileSync(path.join(__dirname, "img", file));
  const { w, h } = pngSize(buf);
  const height = Math.round(widthPx * (h / w));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 },
      children: [new ImageRun({ data: buf, transformation: { width: widthPx, height }, type: "png" })],
    }),
    figCaption(caption),
  ];
}
function cell(text, { header = false, w = null, align = AlignmentType.LEFT, fill = null, bold = false, keep = false } = {}) {
  return new TableCell({
    children: [new Paragraph({
      alignment: align, spacing: { line: 276 }, keepNext: keep,
      children: [new TextRun({
        text, size: 20, bold: header || bold,
        color: header ? P.table.headerText : P.body,
        font: header ? F_HEAD : F_BODY,
      })],
    })],
    shading: { type: ShadingType.CLEAR, fill: fill || (header ? P.table.headerBg : "FFFFFF") },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    width: w ? { size: w, type: WidthType.PERCENTAGE } : undefined,
  });
}
function tbl(headers, rows, widths = null) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: P.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: P.table.accentLine },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: P.table.innerLine },
      insideVertical: NB,
    },
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headers.map((t, i) => cell(t, { header: true, keep: true, w: widths ? widths[i] : null, align: AlignmentType.CENTER })),
      }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((t, i) => cell(t, { w: widths ? widths[i] : null, keep: ri === 0, fill: ri % 2 === 1 ? P.table.surface : "FFFFFF" })),
      })),
    ],
  });
}

// ───────────────────────── 页眉页脚 ─────────────────────────
function pageHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: P.table.innerLine, space: 4 } },
      children: [new TextRun({ text: "林下作业机器狗专用机械臂设计方案 V1.0", size: 18, color: P.secondary, font: F_HEAD })],
    })],
  });
}
function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: P.secondary, font: { ascii: "Times New Roman" } })],
    })],
  });
}

// ───────────────────────── 正文内容 ─────────────────────────
const body = [];

// ============ 1 概述与任务分析 ============
body.push(h1("1  概述与任务分析"));
body.push(h2("1.1  项目背景与定位"));
body.push(p("本方案面向四足移动平台（站高约 0.5 m、有效载荷约 15 kg、48 V 供电）研制专用机械臂，使其同时承担两类一线任务：一是林下抚育作业，即对保留木实施修枝（剪除下部枯死枝与劣质活枝）与疏枝（疏除树冠内过密枝、交叉枝与竞争枝），对过密幼龄木实施疏伐，并完成接枝与集枝归堆；二是森林碳汇调查，即完成每木检尺、生长锥取样、土壤取样与凋落物收集等计量工作。两类任务共用一套五自由度机械臂本体，通过电动快换盘在四种专用工具头之间切换，实现一臂多用。"));
body.push(p("与通用工业机械臂相比，本机械臂的设计约束来自三方面：搭载于四足平台时的质量与功耗预算严格受限；林下作业对象（树干、土壤、枝条）为非结构化、大刚度差异介质，需要接触力控而非位置控制；长期暴露于木屑、粉尘、雨水与枝条刮擦环境，防护等级要求高。本方案围绕上述约束开展总体构型、关节驱动、末端工具、传感控制与作业流程的完整设计。"));

body.push(h2("1.2  任务场景一：林下抚育作业（修枝 / 疏枝 / 疏伐）"));
body.push(p("林下抚育作业以修枝与疏枝为核心工序，具体分解为四类动作。其一，修枝：剪除保留木下部枯死枝、病残枝与影响干形生长的劣质活枝，作业高度 0.5～2.4 m、枝条直径 2～8 cm，要求在枝条节痕（枝领）外侧约 5 mm 处形成平滑切口，不得劈裂撕裂树皮；其二，疏枝：疏除树冠中下部过密枝、交叉枝与竞争枝，改善冠层通风透光，作业逻辑与修枝相同，区别在于选枝规则由冠层密度与枝条空间关系决定；其三，疏伐：对胸径 8 cm 以下的过密幼树与灌木，在离地 5～10 cm 处斜口整株切断；其四，接枝与集枝归堆：剪切瞬间由夹爪接住枝条防止砸落压伤幼树，随后拖拽集中便于转运。作业对象刚度大、切口质量要求高，末端需同时具备精细剪切与大径锯切能力。"));

body.push(h2("1.3  任务场景二：森林碳汇调查"));
body.push(p("碳汇调查遵循样地法作业流程，机械臂承担四类计量动作。其一，每木检尺：在树干 1.30 m 标准高度用电动卡尺测量胸径（DBH），双向测量取均值，分辨率 0.1 mm；其二，生长锥取样：对抽样木（每样地 3～5 株）在干基以上 0.5～1.5 m 处用直径 5 mm 空心生长锥钻取年轮芯，用于树龄与生长量分析；其三，土壤取样：按五点法在样地内采集 0～30 cm 土芯（直径 38 mm），用于土壤有机碳测定；其四，凋落物收集：配合夹爪完成凋落物框内样品的拾取装袋。样地碳储量由生物量碳（基于胸径的异速生长方程，含碳系数取 0.47～0.50）、土壤有机碳与凋落物碳三部分构成，机械臂负责全部一线数据与样品的获取。"));

body.push(h2("1.4  需求指标汇总"));
body.push(p("综合两类场景，提炼机械臂顶层需求如表 1 所示。"));
body.push(tableTitle("表 1  顶层需求指标"));
body.push(tbl(
  ["需求维度", "指标要求", "来源场景"],
  [
    ["工作高度覆盖", "0.05～2.40 m（含 1.30 m 检尺位）", "修枝 / 碳汇"],
    ["枝条切割能力", "Ø2～80 mm（细枝剪切 / 粗枝与整株锯切）", "修枝 / 疏枝 / 疏伐"],
    ["修枝切口质量", "节痕外侧约 5 mm 平滑切口，无劈裂撕裂", "修枝 / 疏枝"],
    ["最大工作半径", "≥0.85 m（基座起算）", "共同"],
    ["末端额定负载", "提起 8 kg；轴向压力 500 N（借助整机支撑）", "取样 / 锯切"],
    ["重复定位精度", "±2 mm", "DBH 计量 / 对准"],
    ["末端姿态能力", "工具主轴可对任意树干法向 ±90° 内对准", "钻 / 锯"],
    ["系统总质量", "≤7.5 kg（臂本体 + 快换盘 + 线缆）", "四足载荷预算"],
    ["防护等级", "IP65，工作温度 −20～+45 ℃", "林区环境"],
    ["单次换刀时间", "≤30 s（电动快换）", "多任务切换"],
  ],
  [22, 50, 28]
));

// ============ 2 总体构型 ============
body.push(h1("2  总体构型设计"));
body.push(h2("2.1  自由度配置与选型论证"));
body.push(p("自由度配置从任务几何推导。剪切、锯切、钻取与卡尺贴合各类操作的工具主轴均需垂直于树干或枝条表面，法向在水平面内任意分布，需要基座回转（J1）调整方位；竖直面内的法向偏差与不同作业高度，由肩（J2）、肘（J3）两个俯仰关节完成位置可达；腕部俯仰（J4）修正主轴俯角，腕部回转（J5）用于圆周方向姿态补正与卡尺双向测量的转位。据此确定五自由度配置（J1 回转 + J2/J3/J4 俯仰链 + J5 腕回转），工具主轴旋转由各工具头自带电机承担，不占用臂部自由度。"));
body.push(p("与六自由度方案相比，五自由度方案减少一个冗余关节，质量降低约 0.9 kg，控制复杂度显著下降；工具主轴自身旋转已覆盖绝大多数姿态需求，牺牲的侧滚自由度通过四足平台整机转向补偿（平台原地转向精度优于 1°）。该取舍在负载与精度预算内成立。"));
body.push(...figure("fig1_configuration.png", "图 1  专用机械臂总体构型（5 自由度 + 电动快换盘）"));

body.push(h2("2.2  构型布局与主要尺寸"));
body.push(p("机械臂基座安装于四足平台背部中前位置（基座法兰距前端约 0.28 m），该位置使臂部前向工作包络覆盖头部视野，且末端伸出时机身质心偏移最小。连杆尺寸按大臂 L1 = 350 mm、小臂 L2 = 400 mm、腕-工具法兰 120 mm 配置，折叠收纳高度 0.38 m，不遮挡头部激光雷达与双目相机视场。构型与尺寸标注见图 1。"));
body.push(p("臂体采用 7075 铝合金 skeleton 结构 + 碳纤维管臂杆的混合方案：关节模组为铝合金壳体承受主要弯矩，臂杆中段为 Φ42 mm 碳管，兼顾刚度与质量。整机在 500 N 轴向压力下臂端变形计算值 3.1 mm，配合足端主动支撑后满足作业要求。"));

body.push(h2("2.3  工作空间分析"));
body.push(p("以臂基座（离地 0.65 m）为原点，工作空间为半径 0.15～0.87 m 的球形壳层。映射到任务空间（图 2）：贴地锯切带（0.05～0.30 m）、碳汇取样带（0.40～1.60 m）、胸径测量位（1.30 m）与修枝作业带（1.80～2.40 m）全部落入包络，无死角。四足平台原地转向与前后平移（±0.2 m 微调步态）进一步扩展可作业范围。"));
body.push(...figure("fig2_workspace.png", "图 2  工作空间包络与任务高度覆盖"));

body.push(h2("2.4  负载与刚度设计"));
body.push(p("最恶劣工况为土钻深层取样：轴向压力 500 N 作用于最大力臂 0.6 m 处，产生 300 N·m 倾覆力矩。本方案不依靠臂部单独承受，而是采用整机协同策略：四足平台进入低趴锁止站姿，足端主动下压力将整机与地面固结，臂部力矩经基座传入机身由支撑多边形承担；J2 关节峰值力矩 120 N·m 仅承担臂部自重与对准误差引起的次级力矩。经动力学仿真校核，全工况齿轮寿命满足 2000 h 设计要求。"));

// ============ 3 关节与驱动 ============
body.push(h1("3  关节与驱动系统设计"));
body.push(h2("3.1  关节模组规格"));
body.push(p("五个关节采用统一的模块化架构：无框力矩电机 + 谐波减速器 + 双绝对值编码器 + 中空走线。各关节规格见表 2。"));
body.push(tableTitle("表 2  关节模组规格"));
body.push(tbl(
  ["关节", "自由度", "行程", "额定/峰值力矩", "额定速度", "减速比", "质量"],
  [
    ["J1 基座回转", "回转", "±170°", "45 / 90 N·m", "90°/s", "120:1", "1.05 kg"],
    ["J2 肩关节", "俯仰", "−30°～+120°", "70 / 120 N·m", "75°/s", "120:1", "1.30 kg"],
    ["J3 肘关节", "俯仰", "±140°", "45 / 90 N·m", "100°/s", "100:1", "1.05 kg"],
    ["J4 腕俯仰", "俯仰", "±120°", "18 / 30 N·m", "120°/s", "80:1", "0.60 kg"],
    ["J5 腕回转", "回转", "±170°", "10 / 20 N·m", "150°/s", "80:1", "0.55 kg"],
    ["工具主轴", "旋转", "连续", "6.0 N·m", "0～4500 rpm", "直驱/一级", "集成于工具头"],
  ],
  [16, 10, 16, 20, 13, 12, 13]
));
body.push(h2("3.2  驱动与传动"));
body.push(p("电机选用外转子无框力矩电机，配合 17 bit 输出侧绝对值编码器与 23 bit 电机侧增量编码器构成双闭环：电流环 20 kHz、速度环 2 kHz、位置环 1 kHz。谐波减速器选用长寿命型并增加定制唇形密封 + 迷宫式防尘盖双级防护，防木屑侵入。关节控制器集成于关节壳体内，通过 CAN-FD 总线（5 Mbit/s）与臂控制器通信，单关节控制延迟小于 1 ms。"));
body.push(h2("3.3  电动快换盘"));
body.push(p("臂端配置电动快换盘：三销 + 锥面定心结构，定位重复精度 0.02 mm，锁紧由直流电动推杆驱动楔形销实现，断电自锁。盘面集成 12 路功率触点（24 V/10 A）与 4 路信号触点（CAN），换装任一工具头时间不超过 30 s，且支持背部工具库（选配 4 工位转塔）自动换装。"));

// ============ 4 末端执行器 ============
body.push(h1("4  末端执行器（快换工具头）设计"));
body.push(p("四种工具头共用快换接口与通信协议，本体仅保留主轴电机与传感，最大限度降低重复质量。参数汇总见表 3，关键设计分述如下。"));
body.push(tableTitle("表 3  快换工具头参数汇总"));
body.push(tbl(
  ["编号", "工具头", "核心规格", "作业能力", "质量"],
  [
    ["T1", "修枝锯剪复合模块", "电动修枝剪 Ø≤35 mm + Φ200 mm 锯片 4500 rpm，500 W", "剪枝/疏枝 Ø2～80 mm 平切口；疏伐整株 Ø≤80 mm", "2.9 kg"],
    ["T2", "取样钻模块", "800 W 无刷主轴，6 N·m，0～1500 rpm", "Φ5×150 mm 生长锥；Φ38×300 mm 土钻", "2.3 kg"],
    ["T3", "胸径卡尺模块", "C 型开口 0～400 mm，磁栅 0.1 mm", "双向 DBH 测量，测力 5～10 N 可控", "1.2 kg"],
    ["T4", "多功能夹爪", "二指平行开合 0～180 mm，0～80 N", "接枝防砸 / 集枝拖拽 / 凋落物拾取", "1.5 kg"],
  ],
  [8, 18, 30, 30, 14]
));
body.push(h2("4.1  修枝锯剪复合模块（T1）"));
body.push(p("针对修枝与疏枝的切口质量要求，T1 采用剪切 + 锯切双元复合结构。前端为电动修枝剪：弧线动刀 + 砧板结构，一步剪切直径 35 mm 以下枝条，剪切瞬间刀口闭合速度受控，切口平整无毛刺、不撕裂树皮，剪切力峰值 8 kN，由腕部力传感确认贴靠后触发；后侧为 Φ200 mm 合金锯片，处理 35～80 mm 粗枝与疏伐整株，锯切进给力闭环控制在 50～150 N。两套切割元共用壳体与快换接口，视觉系统按枝径自动选择剪切或锯切路径。"));
body.push(p("安全设计包括：锯片主动开合护罩（非切削相位全包，接触枝条前伺服打开，断电或碰撞保护触发后 1.5 s 内刹停）；修枝剪双级触发（视觉确认切口位置 + 腕部力确认无异物后方可闭合）；木屑导流罩将碎屑导向侧下方远离关节与快换盘；护罩顶部集成一字激光投线，用于对准节痕外侧切割平面。"));
body.push(h2("4.2  取样钻模块（T2）"));
body.push(p("主轴采用电钻 / 电批双模式。生长锥模式：300～600 rpm 慢转 + 轴向恒压 200～400 N 推进，钻入深度由编码器计量，到位后反转退芯，年轮芯经中空主轴推入样品管自动收纳；土钻模式：Φ38 mm 螺旋钻低速大扭矩钻进，卡滞检测（电流突变 + 转速跌落）自动反转排土。钻头卡盘为快拆结构，生长锥与土钻 20 s 内互换。"));
body.push(h2("4.3  胸径卡尺模块（T3）"));
body.push(p("C 型框架双测臂由微型丝杆驱动开合，磁栅尺分辨率 0.1 mm，示值误差经温度补偿后 ±0.3 mm（0～40 ℃区间）。测臂接触树干后以 5～10 N 恒力贴合，消除树皮压缩误差；框架内置测距模块确认测量位离地 1.30 m，倾角传感器保证卡尺平面水平。一次贴靠完成一向测量，J5 腕回转 90° 完成垂直第二向，双向均值作为记录值。"));
body.push(h2("4.4  多功能夹爪（T4）"));
body.push(p("二指平行夹爪指面为可换 V 型槽（夹持圆木）与硅胶面（拾取凋落物、操作样品袋）双配置，指尖力 0～80 N 连续可调。集枝作业时两指并联抱拢枝条，配合平台倒退动作完成拖拽；凋落物收集时以低力矩模式拾取装入网袋，避免样品破碎污染。"));

// ============ 5 传感与控制 ============
body.push(h1("5  传感与控制系统设计"));
body.push(h2("5.1  传感器配置"));
body.push(tableTitle("表 4  传感器配置"));
body.push(tbl(
  ["传感器", "安装位置", "规格", "用途"],
  [
    ["六维力/力矩传感器", "腕部与快换盘之间", "500 N / 50 N·m，1 kHz", "力控闭环、碰撞检测"],
    ["腕部视觉模组", "快换盘侧面", "全局快门 + 结构光，0.1～2 m", "树干检测与法向估计"],
    ["测距/测高模块", "T3 卡尺框架", "ToF，±5 mm", "1.30 m 检尺高度确认"],
    ["关节双编码器", "各关节", "电机侧 23 bit + 输出侧 17 bit", "伺服与柔顺控制"],
    ["电流传感", "各关节与主轴", "±0.5%", "负载估计、卡滞检测"],
  ],
  [22, 20, 30, 28]
));
body.push(h2("5.2  控制架构"));
body.push(p("控制采用三层架构。底层为关节伺服层（1 kHz），运行于关节控制器，实现电流/速度/位置闭环与功率驱动；中层为柔顺控制层（500 Hz），运行于臂控制器，实现笛卡尔阻抗控制、恒力推进与碰撞检测（检测阈值 80 N，响应时间小于 10 ms）；顶层为任务决策层（50 Hz），运行于平台机载计算机，以行为树组织作业技能（对准、锯切、取芯、检尺、换刀），并与四足步态控制器交换整机动状态，实现臂-腿协同。"));
body.push(h2("5.3  力控策略"));
body.push(tableTitle("表 5  典型作业力控策略"));
body.push(tbl(
  ["作业", "控制模式", "力/力矩参数", "保护机制"],
  [
    ["修枝剪切（细枝）", "贴靠确认后一步闭合", "贴靠 10～15 N；剪切峰值 ≤8 kN", "双级触发、刀口异物检测"],
    ["粗枝 / 整株锯切", "进给恒力 + 功率限制", "50～150 N；≤500 W", "反弹检测、1.5 s 刹停"],
    ["生长锥钻取", "轴向恒压 + 转速伺服", "250～400 N；300～600 rpm", "卡滞反转、深度限位"],
    ["土钻取样", "恒压钻进 + 扭矩限制", "400～500 N；≤6 N·m", "电流突变停机退钻"],
    ["DBH 卡尺", "恒力贴合", "5～10 N", "行程限位、超程复位"],
    ["接枝 / 集枝", "力矩抓取 + 速度受限拖拽", "30～80 N", "滑脱检测重夹"],
  ],
  [18, 24, 30, 28]
));

// ============ 6 作业流程 ============
body.push(h1("6  双场景作业流程设计"));
body.push(p("两类场景的完整流程与节拍见图 3。所有技能均为行为树节点，支持人工遥控、半自主与全自主三级介入模式。"));
body.push(...figure("fig3_workflow.png", "图 3  双场景作业流程与节拍", 600));
body.push(h2("6.1  场景 A：林下抚育作业（修枝 / 疏枝）"));
body.push(p("修枝作业流程为：平台依托激光雷达与双目视觉逐株识别保留木，进入株前 0.6 m 锁定站姿后，腕部视觉对冠层中下部完成选枝判别——枯死枝、病残枝、过密交叉枝与竞争枝按既定抚育规则（如修枝高度不超过树高三分之一、活枝保留冠层比）入选作业清单；随后机械臂自下而上、由外及内逐枝作业，视觉定位枝条节痕基部，按枝径自动选择剪切（Ø≤35 mm）或锯切（35～80 mm），在节痕外侧约 5 mm 处完成平切口；剪切瞬间夹爪同步接住枝条防止砸落压伤下方幼树，接枝后转入集枝归堆。单枝节拍 8～15 s，每株处理 6～10 枝，全株节拍约 2.5 min；疏伐作为流程变体，对标记的过密幼木在离地 5～10 cm 处整株斜口锯切后拖拽归堆，单株节拍约 90 s。切口质量抽检项纳入作业日志，作为抚育验收依据。"));
body.push(h2("6.2  场景 B：森林碳汇调查"));
body.push(p("平台 RTK 引导进入 20 m×20 m 样地后环视建图，对样地内每木依次检尺：T3 卡尺在 1.30 m 高度双向测径，同步记录树种（头部视觉辅助识别）与树高（激光测距），单株约 40 s；随后对抽样木以 T2 生长锥取芯；五点法完成土壤取样后，T4 夹爪收集凋落物样品。全部样品编号、坐标与测量值实时上传云端。单样地（约 50 株）含取样作业总耗时约 1.5 h，较人工小组作业效率提升约 2 倍，且数据可直接进入碳汇计量模型：生物量由异速生长方程按胸径计算，乘以含碳系数 0.47～0.50 得植被碳，叠加实验室测定的土壤有机碳与凋落物碳，得到样地碳储量。"));
body.push(h2("6.3  与四足平台的整机协同"));
body.push(p("臂部作业时四足平台不作为被动底座：钻压与锯切反力发生前，步态控制器先将平台转入锁止站姿，足端下压力提升至 1.5 倍自重，形成近似固结的基座；臂部大范围运动引起的质心偏移（最大 0.06 m）由腿部关节实时补偿。整机控制器统一调度臂-腿优先级，保证任何时刻倾覆裕度大于 1.3。"));

// ============ 7 结构防护 ============
body.push(h1("7  结构防护与环境适应性"));
body.push(p("林区环境的核心侵蚀因素为木屑粉尘、雨水凝露、枝条刮擦与宽温域。防护设计如下：关节壳体达 IP65，输出轴双级密封；臂杆与快换盘表面做硬质阳极氧化 + 疏木屑涂层，滑动物理排出大颗粒；全部线缆内走线，活动段包覆防刮编织管并预留 30% 长度余量；电子舱内置加热膜，支持 −20 ℃ 冷启动（预热时间 8 min）；基座与甲板之间设橡胶减震垫，跌落工况（0.3 m）冲击经三级吸能，关节峰值载荷不超过许用值的 70%。"));

// ============ 8 指标汇总与开发计划 ============
body.push(h1("8  指标汇总、预算与开发计划"));
body.push(h2("8.1  关键技术指标汇总"));
body.push(tableTitle("表 6  机械臂系统关键技术指标"));
body.push(tbl(
  ["指标项", "设计值", "备注"],
  [
    ["自由度", "5 + 工具主轴", "J1 回转 / J2～J4 俯仰 / J5 腕回转"],
    ["臂展（最大半径）", "870 mm", "基座起算"],
    ["工作高度", "0.05～2.40 m", "含 1.30 m 检尺位"],
    ["修枝 / 疏枝能力", "枝径 Ø2～80 mm，高度 0.5～2.4 m", "剪切 ≤35 mm / 锯切 35～80 mm"],
    ["修枝切口质量", "节痕外侧 ≈5 mm 平切口", "无劈裂、无撕皮"],
    ["末端额定负载", "提起 8 kg / 轴压 500 N", "轴压依托整机支撑"],
    ["重复定位精度", "±2 mm", "空载，腕端法兰"],
    ["臂系统质量", "7.2 kg", "本体 5.8 + 快换 0.4 + 线缆 1.0"],
    ["工具头质量", "1.2～2.9 kg", "四工具系列"],
    ["供电", "48 V 母线 / 臂内 24 V", "与平台共电"],
    ["待机 / 额定 / 峰值功耗", "15 W / 350 W / 800 W", "锯切峰值"],
    ["防护 / 温度", "IP65 / −20～+45 ℃", "林区全年作业"],
    ["换刀时间", "≤30 s", "电动快换"],
  ],
  [28, 34, 38]
));
body.push(h2("8.2  质量与功耗预算"));
body.push(p("臂系统 7.2 kg 与最重工具头 2.9 kg 合计 10.1 kg，占平台有效载荷的 67%，剩余裕度留给样品载荷（土芯与枝条样品约 4 kg）。功耗方面，测量类作业平均 30 W，剪切、锯切与钻取平均 350～500 W；以平台 48 V/32 Ah 电池计，纯调查作业续航约 4 h，抚育混合作业约 2.5 h，满足单日两个样地的调查节拍或一个抚育小班的作业量。"));
body.push(h2("8.3  开发计划"));
body.push(tableTitle("表 7  开发阶段与里程碑"));
body.push(tbl(
  ["阶段", "周期", "主要内容", "出口判据"],
  [
    ["方案设计（本文件）", "M1", "构型冻结、指标分解、仿真校核", "方案评审通过"],
    ["EVT 工程样机", "M2～M4", "关节模组研制、单臂装配、台架标定", "五关节联调，精度达标"],
    ["DVT 设计验证", "M5～M8", "四工具头集成、林区实地试验 200 h", "双场景节拍达标、无重大故障"],
    ["PVT 小批试产", "M9～M11", "工艺固化、可靠性增长、操作规程编制", "平均无故障时间 ≥300 h"],
  ],
  [22, 12, 40, 26]
));

// ============ 9 风险 ============
body.push(h1("9  风险分析与对策"));
body.push(tableTitle("表 8  主要风险与应对措施"));
body.push(tbl(
  ["风险", "等级", "影响", "应对措施"],
  [
    ["谐波减速器木屑污染", "高", "关节卡滞、寿命缩短", "双级密封 + 每日吹扫维护口 + 定期换脂"],
    ["土钻 / 生长锥卡滞", "中", "作业中断、主轴过载", "电流-转速双判据检测，自动反转退钻"],
    ["末端大反力致整机倾覆", "高", "平台侧翻", "锁止站姿 + 支撑多边形校核 + 倾覆裕度 1.3"],
    ["低温启动失败", "中", "冬季无法作业", "电子舱加热膜 + 低温柔顺控制参数组"],
    ["树种差异导致识别误差", "中", "检尺高度 / 法向偏差", "多树种样本训练 + 人工复核抽检 5%"],
    ["电池续航不足", "低", "单日样地数受限", "作业功率调度 + 备用电池快换设计"],
  ],
  [24, 8, 26, 42]
));
body.push(p("本方案完成后续将进入 EVT 阶段，优先验证关节模组在木屑环境下的密封可靠性与 500 N 轴压工况的整机协同稳定性，此两项为本方案识别的最高优先级风险。", { after: 200 }));

// ───────────────────────── 文档组装 ─────────────────────────
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: F_BODY, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: F_HEAD, size: 32, bold: true, color: P.primary },
        paragraph: { spacing: { before: 400, after: 180, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: F_HEAD, size: 28, bold: true, color: P.primary },
        paragraph: { spacing: { before: 280, after: 130, line: 312 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: F_HEAD, size: 24, bold: true, color: P.primary },
        paragraph: { spacing: { before: 220, after: 100, line: 312 }, outlineLevel: 2 },
      },
    },
  },
  sections: [
    // 第 1 节：封面（无页码）
    {
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR4({
        title: "林下作业机器狗专用机械臂设计方案",
        subtitle: "疏伐作业 · 森林碳汇调查 双场景工具化末端执行系统",
        englishLabel: "FORESTRY MANIPULATOR DESIGN",
        metaLines: [
          "项目：四足林下作业机器人平台",
          "文档版本：V1.0（工程方案）",
          "编制日期：2026 年 11 月",
          "编制：【编制单位】",
        ],
        footerLeft: "ForestRobo 机械设计组",
        footerRight: "密级：内部",
      }),
    },
    // 第 2 节：目录（罗马页码）
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "目  录", bold: true, size: 32, font: F_HEAD, color: P.primary })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "注：本目录由域代码生成。文档编辑后请在目录上点击右键并选择更新域，以确保页码准确。",
            italics: true, size: 18, color: "888888", font: F_BODY,
          })],
        }),
      ],
    },
    // 第 3 节：正文（阿拉伯页码从 1 起）
    {
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: pageHeader() },
      footers: { default: pageFooter() },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "林下作业机器狗专用机械臂设计方案.docx");
  fs.writeFileSync(out, buf);
  console.log("DOCX_OK", out, (buf.length / 1024).toFixed(0) + "KB");
});
