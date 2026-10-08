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

        // Reset KG panel
        document.getElementById('kgPanel').classList.add('hidden');
        clearKnowledgeGraph();

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
                            throw e;
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

        // Collect all graphs from all claims
        let allGraphNodes = [];
        let allGraphEdges = [];

        report.claims.forEach((item, index) => {
            const span = document.createElement('span');
            span.textContent = item.claim + " ";
            span.className = "claim-span ";

            if (item.verification.label === "contradicted") {
                span.classList.add("claim-hallucinated");
                span.addEventListener('mouseenter', (e) => showTooltip(e, item));
                span.addEventListener('mouseleave', () => startTooltipHideTimer());
                if (item.source_url) {
                    span.addEventListener('click', () => window.open(item.source_url.split(',')[0].trim(), '_blank'));
                }
            } else if (item.verification.label === "supported") {
                span.classList.add("claim-supported");
            } else {
                span.classList.add("claim-unknown");
                span.addEventListener('mouseenter', (e) => showTooltip(e, item));
                span.addEventListener('mouseleave', () => startTooltipHideTimer());
                if (item.source_url) {
                    span.addEventListener('click', () => window.open(item.source_url.split(',')[0].trim(), '_blank'));
                }
            }

            annotatedContainer.appendChild(span);

            // Accumulate graph data
            if (item.graph) {
                if (item.graph.nodes) allGraphNodes.push(...item.graph.nodes);
                if (item.graph.edges) allGraphEdges.push(...item.graph.edges);
            }
        });

        resultsSection.classList.remove('hidden');

        // Render Knowledge Graph if we have any nodes
        const uniqueNodes = deduplicateById(allGraphNodes);
        const uniqueEdges = deduplicateEdges(allGraphEdges);

        if (uniqueNodes.length > 0) {
            document.getElementById('kgPanel').classList.remove('hidden');
            document.getElementById('kgSubtitle').textContent =
                `${uniqueNodes.length} entities, ${uniqueEdges.length} relationships discovered`;
            renderKnowledgeGraph(uniqueNodes, uniqueEdges);
        }
    }

    // ==========================================
    //   KNOWLEDGE GRAPH VISUALIZER (D3.js)
    // ==========================================

    function clearKnowledgeGraph() {
        const svg = document.getElementById('kgSvg');
        if (svg) svg.innerHTML = '';
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

    function renderKnowledgeGraph(nodes, edges) {
        const container = document.getElementById('kgContainer');
        const width = container.clientWidth || 700;
        const height = 380;

        const svg = d3.select("#kgSvg")
            .attr("width", width)
            .attr("height", height);

        svg.selectAll("*").remove(); // clear old graph

        // Arrow marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 28)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#6366f1")
            .style("stroke", "none");

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(55));

        // Draw edges
        const link = svg.append("g")
            .selectAll("line")
            .data(edges)
            .enter().append("line")
            .attr("stroke", "#6366f1")
            .attr("stroke-opacity", 0.5)
            .attr("stroke-width", 1.5)
            .attr("marker-end", "url(#arrowhead)");

        // Edge labels
        const edgeLabel = svg.append("g")
            .selectAll("text")
            .data(edges)
            .enter().append("text")
            .attr("font-size", "9px")
            .attr("fill", "#a5b4fc")
            .attr("text-anchor", "middle")
            .text(d => {
                const rel = d.relationship || "";
                return rel.length > 35 ? rel.substring(0, 33) + "…" : rel;
            });

        // Draw nodes
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
            .attr("r", 28)
            .attr("fill", "rgba(99, 102, 241, 0.15)")
            .attr("stroke", "#818cf8")
            .attr("stroke-width", 2);

        nodeGroup.append("text")
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", "9px")
            .attr("font-weight", "600")
            .attr("fill", "#e0e7ff")
            .text(d => {
                const label = d.label || d.id;
                return label.length > 14 ? label.substring(0, 13) + "…" : label;
            });

        // Tooltip on node hover
        const kgTooltip = d3.select("body").append("div")
            .attr("class", "kg-node-tooltip")
            .style("opacity", 0)
            .style("position", "absolute")
            .style("background", "rgba(15,15,35,0.95)")
            .style("border", "1px solid #6366f1")
            .style("border-radius", "8px")
            .style("padding", "8px 12px")
            .style("font-size", "11px")
            .style("color", "#e0e7ff")
            .style("max-width", "260px")
            .style("pointer-events", "none")
            .style("z-index", "9999");

        nodeGroup
            .on("mouseover", (event, d) => {
                kgTooltip.transition().duration(150).style("opacity", 1);
                kgTooltip.html(`<strong>${d.label}</strong><br>${d.summary_snippet || ""}`)
                    .style("left", (event.pageX + 14) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mousemove", (event) => {
                kgTooltip.style("left", (event.pageX + 14) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", () => {
                kgTooltip.transition().duration(200).style("opacity", 0);
            });

        // Update positions on each simulation tick
        simulation.on("tick", () => {
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

    // ==========================================
    //   TOOLTIP INTERACTION LOGIC
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
        // Show only the first 400 chars of evidence to keep tooltip clean
        const ev = data.evidence || "";
        document.getElementById('ttEvidence').textContent = `"...${ev.substring(0, 400)}..."`;

        const sourceLink = document.getElementById('ttSource');
        if (data.source_url) {
            sourceLink.href = data.source_url.split(',')[0].trim();
            sourceLink.style.display = 'inline-flex';
        } else {
            sourceLink.style.display = 'none';
        }

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
        }, 600);
    }
});
