document.addEventListener('DOMContentLoaded', () => {
    const solSelect = document.getElementById('solSelect');
    const container = document.getElementById('mynetwork');
    const totalWDisp = document.getElementById('totalW');
    const matchCountDisp = document.getElementById('matchCount');
    const pathList = document.getElementById('pathList');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const btnFit = document.getElementById('btnFit');
    
    let network = null;
    let nodesDataset = new vis.DataSet();
    let edgesDataset = new vis.DataSet();
    let allMatches = [];

    const options = {
        nodes: {
            shape: 'dot',
            size: 20,
            font: { size: 14, face: 'Inter', color: '#f0f0f5' },
            borderWidth: 2,
            shadow: { enabled: true, color: 'rgba(0,0,0,0.5)', size: 5 }
        },
        edges: {
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
            smooth: { enabled: true, type: 'curvedCW', roundness: 0.2 }
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            stabilization: { iterations: 150 }
        },
        interaction: { hover: true, tooltipDelay: 200 }
    };

    function initNetwork() {
        if (network) network.destroy();
        const data = { nodes: nodesDataset, edges: edgesDataset };
        network = new vis.Network(container, data, options);
        
        network.on("stabilizationIterationsDone", () => {
            loadingOverlay.classList.remove('active');
            network.setOptions({ physics: { enabled: false } });
        });
    }

    // Color mapper (replicated from explorer for consistency)
    function getColorByOutDegree(degree) {
        if (degree === 0) return { background: '#f6e58d', border: '#eebb22' }; 
        if (degree < 3) return { background: '#badc58', border: '#6ab04c' }; 
        if (degree < 6) return { background: '#7ed6df', border: '#22a6b3' };
        if (degree < 10) return { background: '#686de0', border: '#4834d4' }; 
        return { background: '#a29bfe', border: '#6c5ce7' }; 
    }

    async function loadSolution(filename) {
        try {
            loadingOverlay.classList.add('active');
            const response = await fetch(`/api/solution_data?file=${encodeURIComponent(filename)}`);
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);

            totalWDisp.textContent = data.solution_meta.total_w.toFixed(2);
            matchCountDisp.textContent = data.solution_meta.num_matches;
            allMatches = data.solution_meta.matches;

            // Apply consistent node coloring
            data.nodes.forEach(node => {
                let colors = getColorByOutDegree(node.group);
                if (node.is_altruistic) colors = { background: '#ffffff', border: '#aaaaaa' };

                node.color = {
                    background: colors.background,
                    border: node.borderWidth ? '#ffffff' : colors.border,
                    highlight: { background: '#fff', border: '#ffffff' },
                    hover: { background: '#fff', border: '#ffffff' }
                };
            });

            // Render Path Cards
            pathList.innerHTML = '';
            allMatches.forEach((match, idx) => {
                const card = document.createElement('div');
                card.className = 'path-card';
                card.innerHTML = `
                    <div class="path-header">
                        <span class="type-badge badge-${match.type}">${match.type}</span>
                        <span>W: ${match.predicted_w.toFixed(2)}</span>
                    </div>
                    <div class="path-nodes">${match.node_ids.join(' → ')}</div>
                `;
                card.onclick = () => highlightPath(match, card);
                pathList.appendChild(card);
            });

            nodesDataset.clear();
            edgesDataset.clear();
            nodesDataset.add(data.nodes);
            edgesDataset.add(data.edges);
            
            initNetwork();
        } catch (err) {
            console.error(err);
            loadingOverlay.classList.remove('active');
            alert("Error loading solution: " + err.message);
        }
    }

    function highlightPath(match, cardElement) {
        // Reset all cards
        document.querySelectorAll('.path-card').forEach(c => c.classList.remove('active'));
        cardElement.classList.add('active');

        // Use white for highlighting solution edges as per user request
        const highlightColor = '#ffffff'; 
        const nodeIds = match.node_ids.map(id => String(id));
        
        // Build edge IDs to highlight
        const activeEdges = [];
        if (match.type === 'cycle') {
            for (let i = 0; i < nodeIds.length; i++) {
                activeEdges.push([nodeIds[i], nodeIds[(i + 1) % nodeIds.length]]);
            }
        } else {
            for (let i = 0; i < nodeIds.length - 1; i++) {
                activeEdges.push([nodeIds[i], nodeIds[i + 1]]);
            }
        }

        // Apply visual updates to vis.js
        const allNodes = nodesDataset.get();
        allNodes.forEach(node => {
            const isInPath = nodeIds.includes(String(node.id));
            if (isInPath) {
                node.opacity = 1.0;
                node.borderWidth = 4;
            } else {
                node.opacity = 0.2;
                node.borderWidth = 2;
            }
        });
        nodesDataset.update(allNodes);

        const allEdges = edgesDataset.get();
        allEdges.forEach(edge => {
            const isPart = activeEdges.some(ae => ae[0] === String(edge.from) && ae[1] === String(edge.to));
            if (isPart) {
                edge.color = { color: highlightColor, opacity: 1.0 };
                edge.width = 6;
            } else {
                edge.color = { color: '#ffffff', opacity: 0.1 };
                edge.width = 2;
            }
        });
        edgesDataset.update(allEdges);
        
        // Focus on the path
        network.fit({ nodes: nodeIds, animation: true });
    }

    solSelect.onchange = (e) => loadSolution(e.target.value);
    btnFit.onclick = () => network.fit({ animation: true });

    if (solSelect.value) loadSolution(solSelect.value);
});
