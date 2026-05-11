#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const rows = fs
  .readFileSync(path.join(root, "results.tsv"), "utf8")
  .trim()
  .split(/\n/)
  .slice(1)
  .map((line, index) => {
    const [commit, suite, mean, max, bpb, memory, status, ...desc] = line.split("\t");
    return {
      index,
      commit,
      suite,
      mean: Number(mean),
      max: Number(max),
      bpb: Number(bpb),
      memory: Number(memory),
      status,
      description: desc.join("\t"),
    };
  });

const coreSuites = [
  { key: "core-100k", label: "core-100k", color: "#2563eb" },
  { key: "core-1m", label: "core-1m", color: "#059669" },
  { key: "core-10m-3pair", label: "core-10m-3pair", color: "#dc2626" },
];

const targetedSuites = [
  {
    key: "gemma-o200k-1m",
    label: "Gemma -> o200k 1M",
    color: "#7c3aed",
    match: (suite) =>
      suite === "gemma-o200k-1m-unshuffled-lock192-seed" ||
      suite.startsWith("mainu-gemma-o200k-1m-"),
  },
  {
    key: "gemma-o200k-10m",
    label: "Gemma -> o200k 10M",
    color: "#a855f7",
    match: (suite) =>
      suite === "gemma-o200k-10m-unshuffled-lock192-seed" ||
      suite.startsWith("mainu-gemma-o200k-10m-"),
  },
  {
    key: "o200k-qwen36-1m",
    label: "o200k -> Qwen36 1M",
    color: "#ea580c",
    match: (suite) =>
      suite === "o200k-qwen36-1m-unshuffled-lock192-seed" ||
      suite.startsWith("mainu-o200k-qwen36-1m-"),
  },
  {
    key: "o200k-qwen36-10m",
    label: "o200k -> Qwen36 10M",
    color: "#f97316",
    match: (suite) =>
      suite === "o200k-qwen36-10m-unshuffled-lock192-seed" ||
      suite.startsWith("mainu-o200k-qwen36-10m-"),
  },
];

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmt(value, digits = 6) {
  return Number(value).toFixed(digits);
}

function pct(first, best) {
  return (((first - best) / first) * 100).toFixed(1);
}

function runningBest(points) {
  const out = [];
  let best = Infinity;
  for (const point of points) {
    if (point.status !== "keep" && point.status !== "diagnostic") continue;
    if (point.mean <= best) {
      best = point.mean;
      out.push({ ...point, best });
    }
  }
  return out;
}

function summarize(groups) {
  return groups.map((group) => {
    const points = rows.filter((row) => group.match ? group.match(row.suite) : row.suite === group.key);
    const accepted = points.filter((row) => row.status === "keep" || row.status === "diagnostic");
    const first = accepted[0];
    const best = accepted.reduce((acc, row) => (row.mean < acc.mean ? row : acc), accepted[0]);
    return { ...group, points, accepted, first, best };
  });
}

const core = summarize(coreSuites);
const targeted = summarize(targetedSuites);

const width = 1400;
const height = 980;
const margin = { left: 86, right: 44 };
const coreBox = { x: 86, y: 86, w: 1270, h: 390 };
const targetBox = { x: 86, y: 560, w: 1270, h: 270 };
const minX = 0;
const maxX = rows.length - 1;
const minY = 0.068;
const maxY = 0.589;

function xScale(index, box) {
  return box.x + ((index - minX) / Math.max(1, maxX - minX)) * box.w;
}

function yScale(value, box, min, max) {
  return box.y + box.h - ((value - min) / (max - min)) * box.h;
}

function grid(box, min, max, yTicks, title) {
  let svg = "";
  svg += `<text x="${box.x}" y="${box.y - 20}" font-size="17" font-weight="700">${escapeXml(title)}</text>\n`;
  for (let i = 0; i < yTicks.length; i++) {
    const value = yTicks[i];
    const y = yScale(value, box, min, max);
    svg += `<line x1="${box.x}" y1="${y.toFixed(1)}" x2="${box.x + box.w}" y2="${y.toFixed(1)}" class="grid"/>\n`;
    svg += `<text x="${box.x - 12}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="tiny">${value.toFixed(3)}</text>\n`;
  }
  for (let i = 0; i <= 7; i++) {
    const index = Math.round(minX + ((maxX - minX) * i) / 7);
    const x = xScale(index, box);
    svg += `<line x1="${x.toFixed(1)}" y1="${box.y}" x2="${x.toFixed(1)}" y2="${box.y + box.h}" class="grid"/>\n`;
    svg += `<text x="${x.toFixed(1)}" y="${box.y + box.h + 22}" text-anchor="middle" class="tiny">${index}</text>\n`;
  }
  svg += `<line x1="${box.x}" y1="${box.y + box.h}" x2="${box.x + box.w}" y2="${box.y + box.h}" class="axis"/>\n`;
  svg += `<line x1="${box.x}" y1="${box.y}" x2="${box.x}" y2="${box.y + box.h}" class="axis"/>\n`;
  return svg;
}

function drawCore() {
  let svg = grid(coreBox, minY, maxY, [0.068, 0.155, 0.242, 0.328, 0.415, 0.502, 0.589], "Core suite running best");
  for (const group of core) {
    for (const point of group.points) {
      const x = xScale(point.index, coreBox);
      const y = yScale(point.mean, coreBox, minY, maxY);
      const cls = point.status === "keep" ? "keep" : "discard";
      svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.8" fill="${group.color}" class="${cls}"><title>${escapeXml(`${point.index} ${point.suite} ${point.status} mean=${fmt(point.mean)}: ${point.description}`)}</title></circle>\n`;
    }
    const bests = runningBest(group.points);
    if (bests.length) {
      svg += `<path d="${bests.map((point, i) => `${i === 0 ? "M" : "L"} ${xScale(point.index, coreBox).toFixed(1)} ${yScale(point.best, coreBox, minY, maxY).toFixed(1)}`).join(" ")}" class="line" stroke="${group.color}"/>\n`;
      for (const point of bests) {
        svg += `<circle cx="${xScale(point.index, coreBox).toFixed(1)}" cy="${yScale(point.best, coreBox, minY, maxY).toFixed(1)}" r="5.0" fill="#fff" stroke="${group.color}" stroke-width="2"><title>${escapeXml(`${group.label} best=${fmt(point.best)} at ${point.commit}: ${point.description}`)}</title></circle>\n`;
      }
      const last = bests[bests.length - 1];
      svg += labelText(xScale(last.index, coreBox), yScale(last.best, coreBox, minY, maxY), `${group.label} ${last.best.toFixed(3)}`, group.color, 13);
    }
  }
  return svg;
}

function drawTargeted() {
  const values = targeted.flatMap((group) => group.points.map((point) => point.mean));
  const localMin = Math.max(0, Math.min(...values) - 0.015);
  const localMax = Math.max(...values) + 0.015;
  const ticks = [];
  for (let i = 0; i <= 5; i++) ticks.push(localMin + ((localMax - localMin) * i) / 5);
  let svg = grid(targetBox, localMin, localMax, ticks, "Targeted main-unshuffled screens");
  for (const group of targeted) {
    for (const point of group.points) {
      const x = xScale(point.index, targetBox);
      const y = yScale(point.mean, targetBox, localMin, localMax);
      const cls = point.status === "keep" ? "keep" : "discard";
      svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.0" fill="${group.color}" class="${cls}"><title>${escapeXml(`${point.index} ${point.suite} ${point.status} mean=${fmt(point.mean)}: ${point.description}`)}</title></circle>\n`;
    }
    const bests = runningBest(group.points);
    if (bests.length) {
      svg += `<path d="${bests.map((point, i) => `${i === 0 ? "M" : "L"} ${xScale(point.index, targetBox).toFixed(1)} ${yScale(point.best, targetBox, localMin, localMax).toFixed(1)}`).join(" ")}" class="line" stroke="${group.color}"/>\n`;
      for (const point of bests) {
        svg += `<circle cx="${xScale(point.index, targetBox).toFixed(1)}" cy="${yScale(point.best, targetBox, localMin, localMax).toFixed(1)}" r="5.2" fill="#fff" stroke="${group.color}" stroke-width="2"><title>${escapeXml(`${group.label} best=${fmt(point.best)} at ${point.commit}: ${point.description}`)}</title></circle>\n`;
      }
      const last = bests[bests.length - 1];
      svg += labelText(xScale(last.index, targetBox), yScale(last.best, targetBox, localMin, localMax), `${group.label} ${last.best.toFixed(3)}`, group.color, 12);
    }
  }
  return svg;
}

function labelText(x, y, text, color, size) {
  const anchor = x > width - 230 ? "end" : "start";
  const tx = anchor === "end" ? x - 9 : x + 9;
  return `<text x="${tx.toFixed(1)}" y="${(y - 8).toFixed(1)}" text-anchor="${anchor}" font-size="${size}" font-weight="700" fill="${color}">${escapeXml(text)}</text>\n`;
}

function legend() {
  const entries = [...core, ...targeted];
  let svg = "";
  let x = 86;
  let y = 880;
  for (const group of entries) {
    svg += `<line x1="${x}" y1="${y}" x2="${x + 30}" y2="${y}" stroke="${group.color}" stroke-width="3.2" stroke-linecap="round"/>\n`;
    svg += `<circle cx="${x + 15}" cy="${y}" r="5" fill="#fff" stroke="${group.color}" stroke-width="2"/>\n`;
    svg += `<text x="${x + 40}" y="${y + 5}" class="small">${escapeXml(group.label)}</text>\n`;
    x += group.label.length > 17 ? 245 : 170;
    if (x > 1180) {
      x = 86;
      y += 28;
    }
  }
  svg += `<circle cx="${x}" cy="${y}" r="4" fill="#ef4444" opacity=".45"/>\n`;
  svg += `<text x="${x + 14}" y="${y + 5}" class="small">discarded measured run</text>\n`;
  return svg;
}

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}.small{font-size:13px;fill:#4b5563}.tiny{font-size:11px;fill:#6b7280}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.line{fill:none;stroke-width:3.0;stroke-linejoin:round;stroke-linecap:round}.keep{opacity:.9}.discard{opacity:.28}</style>
<text x="${margin.left}" y="34" font-size="25" font-weight="700">Detokenizer Autoresearch: improvement over time</text>
<text x="${margin.left}" y="58" class="small">Running best CER50k by suite; lower is better. Bottom panel tracks the targeted main-unshuffled Gemma/OpenAI and OpenAI/Qwen sweeps.</text>
${drawCore()}
${drawTargeted()}
<text x="700" y="858" text-anchor="middle" class="small">Experiment log row index</text>
<text x="22" y="281" transform="rotate(-90 22 281)" text-anchor="middle" class="small">core mean CER50k</text>
<text x="22" y="695" transform="rotate(-90 22 695)" text-anchor="middle" class="small">targeted CER50k</text>
${legend()}
<text x="86" y="954" class="tiny">${escapeXml(core.map((group) => `${group.label}: ${group.first.mean.toFixed(3)} -> ${group.best.mean.toFixed(3)} (${pct(group.first.mean, group.best.mean)}% lower)`).join("  |  "))}</text>
</svg>`;

fs.writeFileSync(path.join(root, "plots", "improvement_over_time.svg"), svg);

const coreTable = core
  .map((group) => `| \`${group.label}\` | \`${fmt(group.first.mean)}\` | \`${fmt(group.best.mean)}\` | \`${pct(group.first.mean, group.best.mean)}%\` |`)
  .join("\n");
const targetedTable = targeted
  .map((group) => `| ${group.label} | \`${fmt(group.first.mean)}\` | \`${fmt(group.best.mean)}\` | \`${pct(group.first.mean, group.best.mean)}%\` |`)
  .join("\n");

const markdown = `# Detokenizer Improvement Over Time

![Improvement over time](improvement_over_time.svg)

## Core Suites

| Suite | First accepted | Current best | Relative improvement |
|---|---:|---:|---:|
${coreTable}

## Targeted Main-Unshuffled Screens

| Case | First unshuffled screen | Current best | Relative improvement |
|---|---:|---:|---:|
${targetedTable}
`;

fs.writeFileSync(path.join(root, "plots", "improvement_over_time.md"), markdown);
