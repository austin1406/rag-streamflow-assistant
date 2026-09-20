const form = document.getElementById("query-form");
const input = document.getElementById("question");
const result = document.getElementById("result");

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  const button = form.querySelector("button");
  button.disabled = true;
  result.innerHTML = "<p>Retrieving and generating...</p>";

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    let html = "";
    if (data.error) {
      html += `<div class="answer-box error">${escapeHtml(data.error)}</div>`;
    } else {
      html += `<div class="answer-box">${escapeHtml(data.answer)}</div>`;
    }

    html += '<div class="sources"><h3>Retrieved chunks</h3>';
    data.chunks.forEach((c, i) => {
      html += `<div class="chunk">
        <div class="meta">[${i + 1}] ${escapeHtml(c.source)} — ${escapeHtml(c.loc)} (distance ${c.distance.toFixed(3)})</div>
        <div class="text">${escapeHtml(c.text.slice(0, 400))}${c.text.length > 400 ? "..." : ""}</div>
      </div>`;
    });
    html += "</div>";

    result.innerHTML = html;
  } catch (err) {
    result.innerHTML = `<div class="answer-box error">${escapeHtml(String(err))}</div>`;
  } finally {
    button.disabled = false;
  }
});
