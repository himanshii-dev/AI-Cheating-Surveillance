/**
 * script.js
 * ---------
 * AI Cheating Surveillance Dashboard — client-side logic.
 *
 * Polls /state every 500 ms and updates all dashboard widgets.
 * Fetches violation history from /history when the user clicks "Refresh".
 */

"use strict";

// ── DOM references ──────────────────────────────────────────────────────────
const clockEl           = document.getElementById("clock");
const gaugeBar          = document.getElementById("gauge-bar");
const scoreValue        = document.getElementById("score-value");
const alertBanner       = document.getElementById("alert-banner");

const headIcon          = document.getElementById("head-icon");
const headText          = document.getElementById("head-text");
const eyeIcon           = document.getElementById("eye-icon");
const eyeText           = document.getElementById("eye-text");
const phoneIcon         = document.getElementById("phone-icon");
const phoneText         = document.getElementById("phone-text");
const faceIcon          = document.getElementById("face-icon");
const faceText          = document.getElementById("face-text");

const activeViolations  = document.getElementById("active-violations");
const violationTbody    = document.getElementById("violation-tbody");

// Alert threshold is embedded by Flask into the HTML; we parse it here.
const ALERT_THRESHOLD   = parseInt(
  document.querySelector(".score-max").textContent, 10
) || 15;

// ── Clock ───────────────────────────────────────────────────────────────────
function updateClock() {
  const now  = new Date();
  const pad  = (n) => String(n).padStart(2, "0");
  clockEl.textContent =
    `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Head pose helpers ────────────────────────────────────────────────────────
const HEAD_ICONS = {
  "Looking Left":     "←",
  "Looking Right":    "→",
  "Looking Up":       "↑",
  "Looking Down":     "↓",
  "Looking Straight": "↕",
  "N/A":              "·",
};

function headClass(direction) {
  if (direction === "Looking Straight") return "status-ok";
  if (direction === "N/A")             return "status-neutral";
  return "status-warn";
}

// ── Eye status helpers ───────────────────────────────────────────────────────
const EYE_ICONS = {
  "Looking Center": "◎",
  "Looking Left":   "◁",
  "Looking Right":  "▷",
  "Eyes Closed":    "✕",
  "N/A":            "·",
};

function eyeClass(status) {
  if (status === "Looking Center") return "status-ok";
  if (status === "Eyes Closed")    return "status-danger";
  if (status === "N/A")            return "status-neutral";
  return "status-warn";
}

// ── Score gauge ──────────────────────────────────────────────────────────────
function updateScore(score) {
  const pct = Math.min((score / ALERT_THRESHOLD) * 100, 100);
  gaugeBar.style.width = pct + "%";
  scoreValue.textContent = score;

  if (score === 0) {
    scoreValue.style.color = "var(--ok)";
  } else if (score < ALERT_THRESHOLD * 0.6) {
    scoreValue.style.color = "var(--text-primary)";
  } else if (score < ALERT_THRESHOLD) {
    scoreValue.style.color = "var(--warn)";
  } else {
    scoreValue.style.color = "var(--danger)";
  }
}

// ── Active violations pills ───────────────────────────────────────────────────
function renderPills(violations) {
  if (!violations || violations.length === 0) {
    activeViolations.innerHTML = '<span class="pill pill-ok">All Clear</span>';
    return;
  }
  activeViolations.innerHTML = violations
    .map((v) => {
      let cls = "pill-warn";
      if (v.includes("Phone") || v.includes("No Face") || v.includes("Multiple"))
        cls = "pill-danger";
      return `<span class="pill ${cls}">${v}</span>`;
    })
    .join("");
}

// ── Violation table ──────────────────────────────────────────────────────────
function scoreBadge(score) {
  let cls = "score-low";
  if (score >= ALERT_THRESHOLD)     cls = "score-high";
  else if (score >= ALERT_THRESHOLD * 0.5) cls = "score-mid";
  return `<span class="score-badge ${cls}">${score}</span>`;
}

function vtypeClass(vtype) {
  if (vtype.includes("Phone"))    return "vtype-phone";
  if (vtype.includes("Head"))     return "vtype-head";
  if (vtype.includes("Eye"))      return "vtype-eye";
  if (vtype.includes("No Face"))  return "vtype-noface";
  if (vtype.includes("Multiple")) return "vtype-multi";
  return "";
}

function renderViolationRows(rows) {
  if (!rows || rows.length === 0) {
    violationTbody.innerHTML =
      '<tr><td colspan="4" class="empty-row">No violations recorded yet.</td></tr>';
    return;
  }

  violationTbody.innerHTML = rows
    .map(
      (r) => `
      <tr>
        <td>${r.id}</td>
        <td>${r.timestamp}</td>
        <td class="${vtypeClass(r.violation_type)}">${r.violation_type}</td>
        <td>${scoreBadge(r.score)}</td>
      </tr>`
    )
    .join("");
}

// ── /state polling ───────────────────────────────────────────────────────────
async function pollState() {
  try {
    const response = await fetch("/state");
    if (!response.ok) return;
    const s = await response.json();

    // ---- Head pose
    const hDir = s.head_direction || "N/A";
    headIcon.textContent = HEAD_ICONS[hDir] || "·";
    headText.textContent = hDir;
    headText.className   = "status-text " + headClass(hDir);

    // ---- Eye movement
    const eStat = s.eye_status || "N/A";
    eyeIcon.textContent = EYE_ICONS[eStat] || "·";
    eyeText.textContent = eStat;
    eyeText.className   = "status-text " + eyeClass(eStat);

    // ---- Phone
    if (s.phone_detected) {
      phoneIcon.textContent = "📱";
      phoneText.textContent = "Phone Detected!";
      phoneText.className   = "status-text status-danger";
    } else {
      phoneIcon.textContent = "📵";
      phoneText.textContent = "No Phone";
      phoneText.className   = "status-text status-ok";
    }

    // ---- Faces
    const fc = s.face_count;
    if (fc === 0) {
      faceIcon.textContent = "🚫";
      faceText.textContent = "No Face Detected";
      faceText.className   = "status-text status-danger";
    } else if (fc === 1) {
      faceIcon.textContent = "👤";
      faceText.textContent = "1 Face — OK";
      faceText.className   = "status-text status-ok";
    } else {
      faceIcon.textContent = "👥";
      faceText.textContent = `${fc} Faces Detected!`;
      faceText.className   = "status-text status-danger";
    }

    // ---- Score
    updateScore(s.suspicion_score || 0);

    // ---- Alert
    if (s.alert) {
      alertBanner.classList.remove("hidden");
    } else {
      alertBanner.classList.add("hidden");
    }

    // ---- Active violations
    renderPills(s.active_violations);

    // ---- Inline violation table (from recent_violations in state)
    if (s.recent_violations && s.recent_violations.length > 0) {
      renderViolationRows(s.recent_violations);
    }

  } catch (err) {
    console.warn("[Dashboard] State poll failed:", err);
  }
}

// Poll every 500 ms
setInterval(pollState, 500);
pollState(); // immediate first call

// ── /history fetch (manual refresh) ─────────────────────────────────────────
async function fetchHistory() {
  try {
    const response = await fetch("/history");
    if (!response.ok) return;
    const rows = await response.json();
    renderViolationRows(rows);
  } catch (err) {
    console.warn("[Dashboard] History fetch failed:", err);
  }
}

// Expose fetchHistory globally (called by inline onclick in HTML)
window.fetchHistory = fetchHistory;
