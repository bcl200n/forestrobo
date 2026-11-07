# -*- coding: utf-8 -*-
"""TOC 修补准备：列出 TOC 段落结构 + 精确定位标题实际页码"""
import re
import zipfile

import pypdfium2 as pdfium

d = zipfile.ZipFile('林下作业机器狗专用机械臂设计方案.docx').read(
    'word/document.xml').decode('utf-8')

# 1) 所有 TOC 样式段落及其文本
paras = re.findall(r'<w:p\b.*?</w:p>', d, re.S)
print('== TOC 样式段落 ==')
for p_ in paras:
    if re.search(r'w:pStyle w:val="TOC\d"', p_):
        texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p_)
        print(repr(texts))

# 2) 正文标题实际所在 PDF 页（去空格匹配）
pdf = pdfium.PdfDocument('林下作业机器狗专用机械臂设计方案.pdf')
page_txt = []
for i in range(len(pdf)):
    t = pdf[i].get_textpage().get_text_bounded()
    page_txt.append(re.sub(r'\s+', '', t))
pdf.close()

heads = ['1概述与任务分析', '1.1项目背景与定位', '1.2任务场景一', '1.3任务场景二',
         '1.4需求指标汇总', '2总体构型设计', '2.1自由度配置', '2.2构型布局',
         '2.3工作空间分析', '2.4负载与刚度', '3关节与驱动系统设计', '3.1关节模组规格',
         '3.2驱动与传动', '3.3电动快换盘', '4末端执行器', '4.1圆盘锯模块',
         '4.2取样钻模块', '4.3胸径卡尺模块', '4.4多功能夹爪', '5传感与控制系统设计',
         '5.1传感器配置', '5.2控制架构', '5.3力控策略', '6双场景作业流程设计',
         '6.1场景A', '6.2场景B', '6.3与四足平台', '7结构防护', '8指标汇总',
         '8.1关键技术指标', '8.2质量与功耗预算', '8.3开发计划', '9风险分析']
print('== 标题 → PDF 页 → 正文页码 ==')
mapping = {}
for h in heads:
    for i, t in enumerate(page_txt):
        if i >= 3 and h in t:   # 正文从 PDF 第 4 页起
            mapping[h] = i + 1 - 3
            print(f'{h}: PDF第{i + 1}页 -> 正文第{i + 1 - 3}页')
            break
