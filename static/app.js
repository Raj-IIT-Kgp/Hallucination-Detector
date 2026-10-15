document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const inputText = document.getElementById('inputText');
    const spinner = document.querySelector('.spinner');
    const btnText = document.querySelector('.btn-text');
    const resultsSection = document.getElementById('resultsSection');
    const tooltip = document.getElementById('tooltip');
    const progressSection = document.getElementById('progressSection');
    const progressStepText = document.getElementById('progressStepText');
    const terminalStream = document.getElementById('terminalStream');

    // Quick Sample Buttons
    const sampleFactualBtn = document.getElementById('sampleFactualBtn');
    const sampleHallucinatedBtn = document.getElementById('sampleHallucinatedBtn');
    const sampleComplexBtn = document.getElementById('sampleComplexBtn');

    if (sampleFactualBtn) {
        sampleFactualBtn.addEventListener('click', () => {
            inputText.value = "The Eiffel Tower is located in Paris, France. It was designed and built by Gustave Eiffel for the 1889 World's Fair.";
        });
    }

    if (sampleHallucinatedBtn) {
        sampleHallucinatedBtn.addEventListener('click', () => {
            inputText.value = "Albert Einstein was born in 1950 in San Francisco, California, and he won the Nobel Prize in Literature in 1980.";
        });
    }

    if (sampleComplexBtn) {
        sampleComplexBtn.addEventListener('click', () => {
            inputText.value = "The James Webb Space Telescope was launched in December 2021. It orbits around the Sun-Earth L2 Lagrange point.";
        });
    }

    let isTooltipHovered = false;
    let hideTooltipTimeout;

    analyzeBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        if (!text) {
            alert("Please enter text or select a sample statement to verify.");
            return;
        }

        const forceKG = document.getElementById('forceKgToggle').checked;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Verifying...";
        spinner.style.display = "inline-block";
        resultsSection.classList.add('hidden');
        tooltip.classList.remove('visible');

        progressSection.classList.remove('hidden');
        if (progressStepText) progressStepText.textContent = "Initializing pipeline...";
        terminalStream.innerHTML = '<div class="terminal-line">&bull; Connecting to verification pipeline...</div>';

        // Reset Knowledge Graph
        document.getElementById('kgPanel').classList.add('hidden');
        clearKnowledgeGraph();

        try {
            const response = await fetch('/api/analyze_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, forceKG })
            });

            if (!response.ok) {
                throw new Error("Server error: " + response.statusText);
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
                            if (progressStepText) {
                                progressStepText.textContent = data.message;
                            }
                            const row = document.createElement('div');
                            row.className = 'terminal-line';
                            row.textContent = '• ' + data.message;
                            terminalStream.appendChild(row);

                            while (terminalStream.children.length > 6) {
                                terminalStream.removeChild(terminalStream.firstChild);
                            }
                        } else if (data.status === 'done') {
                            renderResults(text, data);
                        }
                    } catch (e) {
                        console.error("Error parsing stream chunk:", e, line);
                        if (e.message !== "Unexpected end of JSON input") {
                            throw e;
                        }
                    }
                }
            }

        } catch (error) {
            alert("Error running verification: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = "Verify Claims";
            spinner.style.display = "none";
            progressSection.classList.add('hidden');
        }
    });

    function renderResults(originalText, responseData) {
        const report = responseData.report;

        // Update Metric Counters
        document.getElementById('totalClaims').textContent = report.summary.total_claims;
        document.getElementById('hallucinatedClaims').textContent = report.summary.contradicted;
        document.getElementById('supportedClaims').textContent = report.summary.supported;
        
        const confMetric = document.getElementById('conformalGuarantee');
        if (confMetric) {
            confMetric.textContent = (report.summary && report.summary.conformal_coverage_guarantee) 
                ? report.summary.conformal_coverage_guarantee 
                : "90%";
        }

        // Render Annotated Document Spans
        const annotatedContainer = document.getElementById('annotatedText');
        annotatedContainer.innerHTML = '';

        let allGraphNodes = [];
        let allGraphEdges = [];

        report.claims.forEach((item) => {
            const span = document.createElement('span');
            span.textContent = item.claim + " ";
            span.className = "claim-span ";

            const label = item.verification.label;
            if (label === "contradicted") {
                span.classList.add("claim-hallucinated");
            } else if (label === "supported") {
                span.classList.add("claim-supported");
            } else {
                span.classList.add("claim-unknown");
            }

            // Interactive Tooltip Hover on All Verified Claims
            span.addEventListener('mouseenter', (e) => showTooltip(e, item));
            span.addEventListener('mouseleave', () => startTooltipHideTimer());

            if (item.source_url) {
                span.title = "Click to inspect source";
                span.addEventListener('click', () => {
                    window.open(item.source_url.split(',')[0].trim(), '_blank');
                });
            }

            annotatedContainer.appendChild(span);

            // Accumulate Graph Data
            if (item.graph) {
                if (item.graph.nodes) allGraphNodes.push(...item.graph.nodes);
                if (item.graph.edges) allGraphEdges.push(...item.graph.edges);
            }
        });

        resultsSection.classList.remove('hidden');

        // Render Knowledge Graph if Nodes Found
        const uniqueNodes = deduplicateById(allGraphNodes);
        const uniqueEdges = deduplicateEdges(allGraphEdges);

        if (uniqueNodes.length > 0) {
            document.getElementById('kgPanel').classList.remove('hidden');
            document.getElementById('kgSubtitle').textContent =
                `${uniqueNodes.length} entity concepts and ${uniqueEdges.length} semantic relationships mapped • Hover nodes or edges for full details`;
            renderKnowledgeGraph(uniqueNodes, uniqueEdges);
        }
    }

    // ==========================================
    //   KNOWLEDGE GRAPH (Clean Academic D3.js)
    // ==========================================

    function clearKnowledgeGraph() {
        const svg = document.getElementById('kgSvg');
        if (svg) svg.innerHTML = '';
        d3.selectAll(".kg-node-tooltip, .kg-edge-tooltip").remove();
    }

    function deduplicateById(nodes) {
        const seen = new Set();
        return nodes.filter(n => {
            if (seen.has(n.id)) return false;
            seen.add(n.id);
            return true;
        });
    }

    function deduplicateEdges(edges) {
        const seen = new Set();
        return edges.filter(e => {
            const key = [e.source, e.target].sort().join('||');
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function positionKgTooltip(event, tooltipSelection) {
        const tooltipNode = tooltipSelection.node();
        if (!tooltipNode) return;
        const width = tooltipNode.offsetWidth || 320;
        const height = tooltipNode.offsetHeight || 90;

        let left = event.pageX + 14;
        let top = event.pageY - 12;

        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft || 0;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop || 0;

        // Viewport bounds detection
        const maxLeft = scrollLeft + window.innerWidth - width - 16;
        if (left > maxLeft) {
            left = event.pageX - width - 14;
        }
        if (left < scrollLeft + 16) {
            left = scrollLeft + 16;
        }

        const maxTop = scrollTop + window.innerHeight - height - 16;
        if (top > maxTop) {
            top = event.pageY - height - 12;
        }
        if (top < scrollTop + 16) {
            top = scrollTop + 16;
        }

        tooltipSelection
            .style("left", `${left}px`)
            .style("top", `${top}px`);
    }

    function renderKnowledgeGraph(nodes, edges) {
        const container = document.getElementById('kgContainer');
        const width = container.clientWidth || 800;
        const height = 380;

        // Clear existing SVG and lingering tooltips
        d3.selectAll(".kg-node-tooltip, .kg-edge-tooltip").remove();

        const svg = d3.select("#kgSvg")
            .attr("width", width)
            .attr("height", height);

        svg.selectAll("*").remove();

        // Arrow Markers (Verified Blue vs Unverified Slate)
        const defs = svg.append("defs");
        defs.append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 26)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#94a3b8")
            .style("stroke", "none");

        defs.append("marker")
            .attr("id", "arrowhead-verified")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 26)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#2563eb")
            .style("stroke", "none");

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(140))
            .force("charge", d3.forceManyBody().strength(-350))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(50));

        // Tooltip Elements
        const kgTooltip = d3.select("body").append("div")
            .attr("class", "kg-node-tooltip");

        const kgEdgeTooltip = d3.select("body").append("div")
            .attr("class", "kg-edge-tooltip");

        // Helper to generate full rich edge tooltip
        function renderEdgeTooltipHtml(d) {
            const sName = typeof d.source === 'object' ? (d.source.label || d.source.id) : d.source;
            const tName = typeof d.target === 'object' ? (d.target.label || d.target.id) : d.target;
            const rel = d.relationship || "related to";
            const isVer = Boolean(d.verified);
            const quote = d.quote;

            let html = `
                <div class="kg-tooltip-header">
                    <span class="kg-tooltip-badge ${isVer ? 'verified' : 'unverified'}">
                        ${isVer ? '✓ Verified Relation' : 'Extracted Relation'}
                    </span>
                    <span class="kg-tooltip-endpoints">${escapeHtml(sName)} ➔ ${escapeHtml(tName)}</span>
                </div>
                <div class="kg-tooltip-relation">
                    ${escapeHtml(rel)}
                </div>
            `;

            if (quote) {
                html += `
                    <div class="kg-tooltip-quote">
                        <div class="kg-tooltip-quote-label">Evidence Quote</div>
                        <div class="kg-tooltip-quote-text">"${escapeHtml(quote)}"</div>
                    </div>
                `;
            }

            return html;
        }

        // Draw Links
        const linkGroup = svg.append("g").attr("class", "kg-links");

        // Transparent wide hit-area line for effortless hovering
        const linkHit = linkGroup.selectAll(".kg-link-hit")
            .data(edges)
            .enter().append("line")
            .attr("class", "kg-link-hit")
            .attr("stroke", "#000000")
            .attr("stroke-opacity", 0)
            .attr("stroke-width", 24)
            .attr("stroke-linecap", "round")
            .style("pointer-events", "stroke")
            .attr("cursor", "pointer");

        // Visible link line
        const link = linkGroup.selectAll(".kg-link-visible")
            .data(edges)
            .enter().append("line")
            .attr("class", "kg-link-visible")
            .attr("stroke", d => d.verified ? "#2563eb" : "#cbd5e1")
            .attr("stroke-width", d => d.verified ? 2.0 : 1.5)
            .attr("stroke-dasharray", d => d.verified ? "none" : "4,3")
            .attr("marker-end", d => d.verified ? "url(#arrowhead-verified)" : "url(#arrowhead)")
            .attr("cursor", "pointer")
            .style("transition", "stroke 0.15s ease, stroke-width 0.15s ease");

        // Edge Relationship Labels
        const edgeLabelGroup = svg.append("g").attr("class", "kg-edge-labels");

        const edgeLabel = edgeLabelGroup.selectAll(".kg-edge-label")
            .data(edges)
            .enter().append("text")
            .attr("class", "kg-edge-label")
            .attr("font-family", "'Inter', sans-serif")
            .attr("font-size", "10px")
            .attr("font-weight", d => d.verified ? "600" : "500")
            .attr("fill", d => d.verified ? "#1d4ed8" : "#64748b")
            .attr("text-anchor", "middle")
            .attr("cursor", "pointer")
            .style("pointer-events", "all")
            .style("paint-order", "stroke fill")
            .style("stroke", "#f8fafc")
            .style("stroke-width", "5px")
            .style("stroke-linecap", "round")
            .style("stroke-linejoin", "round")
            .style("transition", "fill 0.15s ease, font-weight 0.15s ease")
            .text(d => {
                const prefix = d.verified ? "✓ " : "";
                const rel = d.relationship || "";
                const full = prefix + rel;
                return full.length > 32 ? full.substring(0, 30) + "…" : full;
            });

        // Edge Hover Listeners (attached to hit line, visible line, and edge label)
        function handleEdgeMouseOver(event, d) {
            kgEdgeTooltip.html(renderEdgeTooltipHtml(d));
            kgEdgeTooltip.classed("visible", true);
            positionKgTooltip(event, kgEdgeTooltip);

            // Highlight corresponding edge line
            link.filter(l => l === d)
                .attr("stroke-width", 3.5)
                .attr("stroke", d.verified ? "#1d4ed8" : "#334155");

            // Highlight corresponding label
            edgeLabel.filter(l => l === d)
                .attr("fill", d.verified ? "#1e40af" : "#0f172a")
                .attr("font-weight", "700");

            // Highlight connected source and target nodes
            const sId = d.source.id || d.source;
            const tId = d.target.id || d.target;
            nodeGroup.filter(n => n.id === sId || n.id === tId)
                .select("circle")
                .attr("stroke-width", 3.5)
                .attr("stroke", "#1d4ed8");
        }

        function handleEdgeMouseMove(event, d) {
            positionKgTooltip(event, kgEdgeTooltip);
        }

        function handleEdgeMouseOut(event, d) {
            kgEdgeTooltip.classed("visible", false);

            // Restore edge line
            link.filter(l => l === d)
                .attr("stroke-width", d => d.verified ? 2.0 : 1.5)
                .attr("stroke", d => d.verified ? "#2563eb" : "#cbd5e1");

            // Restore label
            edgeLabel.filter(l => l === d)
                .attr("fill", d => d.verified ? "#1d4ed8" : "#64748b")
                .attr("font-weight", d => d.verified ? "600" : "500");

            // Restore connected nodes
            const sId = d.source.id || d.source;
            const tId = d.target.id || d.target;
            nodeGroup.filter(n => n.id === sId || n.id === tId)
                .select("circle")
                .attr("stroke-width", 2)
                .attr("stroke", "#2563eb");
        }

        linkHit
            .on("mouseover", handleEdgeMouseOver)
            .on("mousemove", handleEdgeMouseMove)
            .on("mouseout", handleEdgeMouseOut);

        link
            .on("mouseover", handleEdgeMouseOver)
            .on("mousemove", handleEdgeMouseMove)
            .on("mouseout", handleEdgeMouseOut);

        edgeLabel
            .on("mouseover", handleEdgeMouseOver)
            .on("mousemove", handleEdgeMouseMove)
            .on("mouseout", handleEdgeMouseOut);

        // Draw Nodes
        const nodeGroup = svg.append("g")
            .selectAll("g")
            .data(nodes)
            .enter().append("g")
            .attr("cursor", "pointer")
            .call(d3.drag()
                .on("start", dragStarted)
                .on("drag", dragged)
                .on("end", dragEnded))
            .on("click", (event, d) => {
                if (d.url && d.url !== "#") window.open(d.url, '_blank');
            });

        nodeGroup.append("circle")
            .attr("r", 25)
            .attr("fill", "#eff6ff")
            .attr("stroke", "#2563eb")
            .attr("stroke-width", 2)
            .style("transition", "stroke 0.15s ease, stroke-width 0.15s ease");

        nodeGroup.append("text")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-family", "'Inter', sans-serif")
            .attr("font-size", "10px")
            .attr("font-weight", "600")
            .attr("fill", "#0f172a")
            .text(d => {
                const label = d.label || d.id;
                return label.length > 13 ? label.substring(0, 12) + "…" : label;
            });

        // Hover Node Listeners
        nodeGroup
            .on("mouseover", (event, d) => {
                let nodeHtml = `
                    <div class="kg-tooltip-header">
                        <span class="kg-tooltip-badge verified">Entity Node</span>
                        <span class="kg-tooltip-endpoints">${escapeHtml(d.label || d.id)}</span>
                    </div>
                    <div class="kg-tooltip-relation" style="font-size: 0.82rem; font-weight: 500; color: #334155;">
                        ${escapeHtml(d.summary_snippet || "Knowledge graph concept")}
                    </div>
                `;
                if (d.url && d.url !== "#") {
                    nodeHtml += `<div style="margin-top: 6px; font-size: 0.72rem; color: #2563eb; font-weight: 600;">
                        Click to view source Wikipedia article ↗
                    </div>`;
                }

                kgTooltip.html(nodeHtml);
                kgTooltip.classed("visible", true);
                positionKgTooltip(event, kgTooltip);

                // Highlight node
                d3.select(event.currentTarget).select("circle")
                    .attr("stroke-width", 3.5)
                    .attr("stroke", "#1d4ed8");

                // Highlight incident edges
                link.filter(l => (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id)
                    .attr("stroke-width", 3.0)
                    .attr("stroke", "#1d4ed8");
            })
            .on("mousemove", (event) => {
                positionKgTooltip(event, kgTooltip);
            })
            .on("mouseout", (event, d) => {
                kgTooltip.classed("visible", false);

                d3.select(event.currentTarget).select("circle")
                    .attr("stroke-width", 2)
                    .attr("stroke", "#2563eb");

                link.filter(l => (l.source.id || l.source) === d.id || (l.target.id || l.target) === d.id)
                    .attr("stroke-width", l => l.verified ? 2.0 : 1.5)
                    .attr("stroke", l => l.verified ? "#2563eb" : "#cbd5e1");
            });

        simulation.on("tick", () => {
            linkHit
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            edgeLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);

            nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        function dragStarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        function dragEnded(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }
    window.renderKnowledgeGraph = renderKnowledgeGraph;

    // ==========================================
    //   TOOLTIP POPOVER LOGIC
    // ==========================================

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

        const badge = document.getElementById('ttBadge');
        const label = data.verification.label.toLowerCase();
        
        if (label === 'supported') {
            badge.textContent = 'VERIFIED';
            badge.style.backgroundColor = 'var(--success-light)';
            badge.style.color = 'var(--success-text)';
            badge.style.border = '1px solid var(--success-border)';
        } else if (label === 'unknown') {
            badge.textContent = 'UNCERTAIN';
            badge.style.backgroundColor = 'var(--warning-light)';
            badge.style.color = 'var(--warning-text)';
            badge.style.border = '1px solid var(--warning-border)';
        } else {
            badge.textContent = 'HALLUCINATION';
            badge.style.backgroundColor = 'var(--danger-light)';
            badge.style.color = 'var(--danger-text)';
            badge.style.border = '1px solid var(--danger-border)';
        }

        const confScore = (data.verification.confidence * 100).toFixed(1);
        document.getElementById('ttConfidence').textContent = `${confScore}% Confidence`;

        // Conformal Prediction Set Chips
        const confContainer = document.getElementById('ttConformalSet');
        if (confContainer) {
            confContainer.innerHTML = '';
            const confSet = data.verification.conformal_set || [data.verification.label];
            confSet.forEach(lbl => {
                const chip = document.createElement('span');
                chip.className = `conformal-chip chip-${lbl}`;
                chip.textContent = lbl.toUpperCase();
                confContainer.appendChild(chip);
            });

            if (data.verification.is_escalation_candidate) {
                const escChip = document.createElement('span');
                escChip.className = 'conformal-chip chip-escalated';
                escChip.textContent = 'ESCALATED';
                confContainer.appendChild(escChip);
            }
        }

        // Conformal p-values
        const pValContainer = document.getElementById('ttPValues');
        if (pValContainer) {
            if (data.verification.conformal_p_values) {
                const pv = data.verification.conformal_p_values;
                pValContainer.textContent = `p(contra): ${pv.contradicted ?? '—'} | p(unk): ${pv.unknown ?? '—'} | p(supp): ${pv.supported ?? '—'}`;
            } else {
                pValContainer.textContent = `Guaranteed Coverage: ${data.verification.coverage_guarantee || '90%'}`;
            }
        }

        // Evidence Quote
        const ev = data.evidence || "No corroborating evidence retrieved.";
        document.getElementById('ttEvidence').textContent = `"${ev.substring(0, 320)}${ev.length > 320 ? '...' : ''}"`;

        // Source Citation Link
        const sourceLink = document.getElementById('ttSource');
        if (data.source_url) {
            sourceLink.href = data.source_url.split(',')[0].trim();
            sourceLink.style.display = 'inline-flex';
        } else {
            sourceLink.style.display = 'none';
        }

        // Tooltip Positioning
        const rect = e.target.getBoundingClientRect();
        const tooltipX = Math.max(16, Math.min(window.innerWidth - 400, rect.left + window.scrollX - 20));
        const tooltipY = rect.bottom + window.scrollY + 8;

        tooltip.style.left = `${tooltipX}px`;
        tooltip.style.top = `${tooltipY}px`;

        tooltip.classList.add('visible');
        tooltip.style.pointerEvents = 'auto';
    }

    function startTooltipHideTimer() {
        hideTooltipTimeout = setTimeout(() => {
            if (!isTooltipHovered) {
                tooltip.classList.remove('visible');
                tooltip.style.pointerEvents = 'none';
            }
        }, 350);
    }
});
