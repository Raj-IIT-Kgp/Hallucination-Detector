document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const inputText = document.getElementById('inputText');
    const spinner = document.querySelector('.spinner');
    const btnText = document.querySelector('.btn-text');
    const resultsSection = document.getElementById('resultsSection');
    const tooltip = document.getElementById('tooltip');
    const progressSection = document.getElementById('progressSection');
    const progressText = document.getElementById('progressText');

    let isTooltipHovered = false;
    let hideTooltipTimeout;

    analyzeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        spinner.style.display = "block";
        resultsSection.classList.add('hidden');
        tooltip.classList.remove('visible');
        
        progressSection.classList.remove('hidden');
        progressText.textContent = "Connecting to agents...";

        try {
            const response = await fetch('/api/analyze_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error("API Request Failed: " + response.statusText);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                
                buffer = lines.pop();
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.status === 'error') {
                            throw new Error(data.message);
                        } else if (data.status === 'progress') {
                            progressText.textContent = data.message;
                        } else if (data.status === 'done') {
                            renderResults(text, data);
                        }
                    } catch (e) {
                        console.error("Error parsing JSON chunk:", e, line);
                        if (e.message !== "Unexpected end of JSON input") {
                            throw e; // rethrow logic errors
                        }
                    }
                }
            }
            
        } catch (error) {
            alert("Error running analysis: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = "Detect Hallucinations";
            spinner.style.display = "none";
            progressSection.classList.add('hidden');
        }
    });

    function renderResults(originalText, responseData) {
        const report = responseData.report;
        
        // Update Metrics
        document.getElementById('totalClaims').textContent = report.summary.total_claims;
        document.getElementById('hallucinatedClaims').textContent = report.summary.contradicted;
        document.getElementById('supportedClaims').textContent = report.summary.supported;

        // Render Annotated Text
        const annotatedContainer = document.getElementById('annotatedText');
        annotatedContainer.innerHTML = '';
        
        let rewrittenOutput = [];

        report.claims.forEach((item, index) => {
            const span = document.createElement('span');
            span.textContent = item.claim + " ";
            span.className = "claim-span ";
            
            if (item.verification.label === "contradicted") {
                span.classList.add("claim-hallucinated");
                
                // Add event listeners for tooltip
                span.addEventListener('mouseenter', (e) => showTooltip(e, item));
                span.addEventListener('mouseleave', () => startTooltipHideTimer());
                
                // Redirect on click
                if (item.source_url) {
                    span.addEventListener('click', () => {
                        window.open(item.source_url, '_blank');
                    });
                }
                
                rewrittenOutput.push(item.correction || item.claim);
            } else if (item.verification.label === "supported") {
                span.classList.add("claim-supported");
                rewrittenOutput.push(item.claim);
            } else {
                // Unknown label
                span.classList.add("claim-unknown");
                
                // Add event listeners for tooltip
                span.addEventListener('mouseenter', (e) => showTooltip(e, item));
                span.addEventListener('mouseleave', () => startTooltipHideTimer());
                
                // Redirect on click
                if (item.source_url) {
                    span.addEventListener('click', () => {
                        window.open(item.source_url, '_blank');
                    });
                }
                
                rewrittenOutput.push(item.claim);
            }
            
            annotatedContainer.appendChild(span);
        });

        // Update Rewritten Text
        document.getElementById('rewrittenText').textContent = rewrittenOutput.join(" ");
        
        resultsSection.classList.remove('hidden');
    }

    // Tooltip Interaction Logic
    tooltip.addEventListener('mouseenter', () => {
        isTooltipHovered = true;
        clearTimeout(hideTooltipTimeout);
    });

    tooltip.addEventListener('mouseleave', () => {
        isTooltipHovered = false;
        tooltip.classList.remove('visible');
        tooltip.style.pointerEvents = 'none';
    });

    function showTooltip(e, data) {
        clearTimeout(hideTooltipTimeout);
        
        const badge = document.querySelector('.status-badge');
        badge.textContent = data.verification.label.toUpperCase();
        if (data.verification.label === 'unknown') {
            badge.style.background = 'rgba(156, 163, 175, 0.2)';
            badge.style.color = '#9ca3af';
        } else {
            badge.style.background = 'rgba(239, 68, 68, 0.2)';
            badge.style.color = 'var(--error)';
        }

        document.getElementById('ttConfidence').textContent = `${(data.verification.confidence * 100).toFixed(1)}% Confidence`;
        document.getElementById('ttCorrection').textContent = data.correction || "No correction available.";
        document.getElementById('ttEvidence').textContent = `"...${data.evidence}..."`;
        
        const sourceLink = document.getElementById('ttSource');
        if (data.source_url) {
            sourceLink.href = data.source_url;
            sourceLink.style.display = 'inline-flex';
        } else {
            sourceLink.style.display = 'none';
        }

        // Position tooltip near the cursor
        const rect = e.target.getBoundingClientRect();
        tooltip.style.left = `${Math.max(10, rect.left + window.scrollX - 50)}px`;
        tooltip.style.top = `${rect.bottom + window.scrollY + 10}px`;
        
        tooltip.classList.add('visible');
        tooltip.style.pointerEvents = 'auto';
    }

    function startTooltipHideTimer() {
        hideTooltipTimeout = setTimeout(() => {
            if (!isTooltipHovered) {
                tooltip.classList.remove('visible');
                tooltip.style.pointerEvents = 'none';
            }
        }, 600); // Increased from 200ms to 600ms to give plenty of time to click the link
    }
});
