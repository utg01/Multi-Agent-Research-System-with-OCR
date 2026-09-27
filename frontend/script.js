// frontend/script.js
const BACKEND_URL = "https://multi-agent-research-system-with-ocr.onrender.com";

const form = document.getElementById("research-form");
const input = document.getElementById("query-input");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const resultTopic = document.getElementById("result-topic");
const planDetails = document.getElementById("plan-details");
const planContent = document.getElementById("plan-content");
const reportContent = document.getElementById("report-content");
const downloadBtn = document.getElementById("download-btn");

let lastReportMarkdown = "";
let researchTimer = null;
let researchStartedAt = 0;

function formatElapsedTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function startResearchTimer() {
    researchStartedAt = Date.now();
    statusEl.textContent = "Researching... 0:00 elapsed. This can take a minute or two.";
    researchTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - researchStartedAt) / 1000);
        statusEl.textContent = `Researching... ${formatElapsedTime(elapsed)} elapsed. This can take a minute or two.`;
    }, 1000);
}

function stopResearchTimer() {
    if (researchTimer) {
        clearInterval(researchTimer);
        researchTimer = null;
    }
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    btnLabel.classList.toggle("hidden", isLoading);
    btnSpinner.classList.toggle("hidden", !isLoading);
}

function renderPlan(plan) {
    planContent.innerHTML = "";
    const groups = [
        ["Web queries", plan.web_queries],
        ["Wikipedia topics", plan.wikipedia_topics],
        ["Paper queries", plan.paper_queries],
    ];

    let hasAny = false;
    groups.forEach(([label, items]) => {
        if (!items || !items.length) return;
        hasAny = true;
        const group = document.createElement("div");
        group.className = "plan-group";
        const items_html = items.map((i) => `<li>${i}</li>`).join("");
        group.innerHTML = `<strong>${label}:</strong><ul>${items_html}</ul>`;
        planContent.appendChild(group);
    });

    planDetails.classList.toggle("hidden", !hasAny);
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    setLoading(true);
    statusEl.classList.remove("hidden", "error");
    startResearchTimer();
    resultEl.classList.add("hidden");

    try {
        const res = await fetch(`${BACKEND_URL}/research`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });

        if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.detail || `Request failed (${res.status})`);
        }

        const data = await res.json();

        lastReportMarkdown = data.report || "";
        resultTopic.textContent = data.topic || query;
        reportContent.innerHTML = marked.parse(lastReportMarkdown);
        renderPlan(data.plan || {});

        resultEl.classList.remove("hidden");
        statusEl.classList.add("hidden");
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.classList.add("error");
    } finally {
        stopResearchTimer();
        setLoading(false);
    }
});

downloadBtn.addEventListener("click", async () => {
    if (!lastReportMarkdown) return;

    const filename = `${(resultTopic.textContent || "research-report").replace(/\s+/g, "_").toLowerCase()}.pdf`;

    try {
        const response = await fetch(`${BACKEND_URL}/export-pdf`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: resultTopic.textContent || "Research report",
                report: lastReportMarkdown,
            }),
        });

        if (!response.ok) {
            const errorBody = await response.json().catch(() => ({}));
            throw new Error(errorBody.detail || `PDF export failed (${response.status})`);
        }

        const blobUrl = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = blobUrl;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(blobUrl);
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.classList.remove("hidden");
        statusEl.classList.add("error");
    }
});