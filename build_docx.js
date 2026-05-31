// Build NitroDQN_app_section.docx — 5 sections, captioned figures.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun,
  HeadingLevel, AlignmentType, PageOrientation,
} = require("docx");

const ROOT = __dirname;
const SHOT = (n) => path.join(ROOT, "screenshots", n);

// US Letter portrait (DXA): content width with 1" margins = 9360 DXA
// Image sizes in pixels (docx-js treats transformation as px, ~96 DPI):
//   full-width ~ 600 px,  narrow LLM panel ~ 260 px (~7cm)
const FULL_W = 600, FULL_H_OF = (origW, origH) => Math.round((FULL_W / origW) * origH);
const NARROW_W = 260, NARROW_H = Math.round((NARROW_W / 290) * 640); // crop is 290x640

function bodyPara(text) {
  return new Paragraph({
    spacing: { after: 160, line: 300 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "Calibri", size: 22 })], // 11pt
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: "Calibri", size: 28, bold: true })],
  });
}

function title(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { after: 320 },
    children: [new TextRun({ text, font: "Calibri", size: 36, bold: true })],
  });
}

function figure(filename, captionText, opts = {}) {
  const data = fs.readFileSync(SHOT(filename));
  const width  = opts.width  ?? FULL_W;
  const height = opts.height ?? Math.round(width * 0.5625);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 80 },
      children: [new ImageRun({
        type: "png",
        data,
        transformation: { width, height },
        altText: { title: captionText, description: captionText, name: filename },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 280 },
      children: [new TextRun({ text: captionText, font: "Calibri", size: 20, italics: true })],
    }),
  ];
}

// ── Section bodies (verbatim from report_section.md §1-5) ───────────────────

const S1 = `The landing view (Figure 1) gives the user a complete picture of the system before any simulation is run. The left column reports the current crop status (cumulative nitrogen, stress, biomass, leaf area index, vegetative stage, rainfall, leached nitrogen). The centre column hosts the DQN controller (episode controls, manual override doses, the live Q-value bar chart, and the season timeline). The right column is reserved for the LLM co-pilot, organised into three tabs (State, Decision, Advisor). The bottom row shows the 1200-episode training history loaded from dqn_training_log.json.`;

const S2 = `Once a season is started, the State tab (Figure 2) converts the raw observation dictionary returned by the DSSAT-compatible environment into a short natural-language summary aimed at a non-technical reader. The text is produced by the LLM service from the current values of nstres, xlai, vstage, cumsumfert, and rain.`;

const S3 = `After each step, the Decision tab (Figure 3) explains why the DQN chose a particular fertilizer dose. The panel reports the chosen action, the Q-value margin over the next-best action (a proxy for confidence), and the dominant state variables that drove the choice. The Q-values for all five actions are listed below the prose explanation so the user can compare the alternatives.`;

const S4 = `The "Auto 10 Steps" and "Run Full Season" buttons advance the simulation under the greedy DQN policy without further input. The season timeline (Figure 4) draws each applied dose as a blue bar and overlays the evolving N-stress factor in red, making the relationship between fertilisation events and stress response visible at a glance. The five dose buttons above the chart let the user override the agent and apply any dose manually, which is useful for comparing the learned policy against simple baselines.`;

const S5 = `The Advisor tab (Figure 5) is a free-form chat interface that takes the current crop state as conversational context. The user can ask agronomic questions ("what does the nstres score mean and when should I worry?") and receive grounded answers from the language model. Conversation history is preserved across turns within the same session.`;

// Get pixel dims for the full-width screenshots so aspect ratio is preserved.
function pngSize(file) {
  const buf = fs.readFileSync(SHOT(file));
  // PNG: width at byte 16-19, height at 20-23 (big-endian)
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  return { w, h };
}
const sz01 = pngSize("01_dashboard_overview.png");
const sz04 = pngSize("04_after_auto10.png");
const h01 = Math.round((FULL_W / sz01.w) * sz01.h);
const h04 = Math.round((FULL_W / sz04.w) * sz04.h);

const children = [
  title("NitroDQN Application"),
  bodyPara("This section describes the web application built to demonstrate the trained DQN agent and its LLM co-pilot. The application is a single-page dashboard served by a FastAPI backend. It exposes the simulation loop, the agent's internal Q-values, and a chat interface to a language model that explains the agent's decisions in plain language."),

  h2("1. Dashboard Overview"),
  bodyPara(S1),
  ...figure("01_dashboard_overview.png", "Figure 1 — Dashboard on first load.",
           { width: FULL_W, height: h01 }),

  h2("2. State Explainer"),
  bodyPara(S2),
  ...figure("02_state_explainer_cropped.png", "Figure 2 — State Explainer panel after starting an episode.",
           { width: NARROW_W, height: NARROW_H }),

  h2("3. Decision Narrator"),
  bodyPara(S3),
  ...figure("03_decision_narrator_cropped.png", "Figure 3 — Decision Narrator after a single DQN step.",
           { width: NARROW_W, height: NARROW_H }),

  h2("4. Season Progress and Manual Override"),
  bodyPara(S4),
  ...figure("04_after_auto10.png", "Figure 4 — Season state after ten automated steps.",
           { width: FULL_W, height: h04 }),

  h2("5. Strategy Advisor (Chat)"),
  bodyPara(S5),
  ...figure("05_advisor_chat_cropped.png", "Figure 5 — Advisor chat with a sample question and reply.",
           { width: NARROW_W, height: NARROW_H }),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840, orientation: PageOrientation.PORTRAIT },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "NitroDQN_app_section.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
