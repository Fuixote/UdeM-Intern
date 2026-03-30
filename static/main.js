document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const batchSelect = document.getElementById('batchSelect');
    const datasetSelect = document.getElementById('datasetSelect');
    const container = document.getElementById('mynetwork');
    const statusBadge = document.getElementById('statusBadge');
    const nodeCountSpan = document.getElementById('nodeCount');
    const edgeCountSpan = document.getElementById('edgeCount');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const btnFit = document.getElementById('btnFit');
    const btnPhysics = document.getElementById('btnPhysics');
    
    // Vis.js instances
    let network = null;
    let nodesDataset = new vis.DataSet();
    let edgesDataset = new vis.DataSet();
    let physicsEnabled = false;
    const datasetConfig = window.datasetConfig || { batch_files: {}, default_batch: null };

        // We use a custom function to transform the title strings into actual HTML elements before giving to Vis.js
        const createTooltipElement = (htmlString) => {
            const container = document.createElement('div');
            container.innerHTML = htmlString;
            return container;
        };
        
        // Setup options
        const options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 40,
                    label: {
                        enabled: true,
                        min: 12,
                        max: 20,
                        maxVisible: 20,
                        drawThreshold: 5
                    }
                },
            font: {
                size: 14,
                face: 'Inter',
                color: '#f0f0f5',
                strokeWidth: 2,
                strokeColor: '#1a1a20'
            },
            borderWidth: 2,
            borderWidthSelected: 4,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.5)',
                size: 10,
                x: 0,
                y: 4
            }
        },
        edges: {
            width: 1, // Default thickness, scaling dynamically
            scaling: {
                min: 1,
                max: 8, // Max thickness for highest score
                label: { enabled: false }
            },
            color: {
                color: 'rgba(108, 92, 231, 0.4)', // Dim-violet default color
                highlight: '#a29bfe',
                hover: '#8172ea',
                opacity: 0.8
            },
            smooth: {
                enabled: true,
                type: 'curvedCW', // Curve edges to avoid overlapping bi-directional paths
                roundness: 0.2
            },
            arrows: {
                to: { enabled: true, scaleFactor: 0.5, type: 'arrow' }
            },
            selectionWidth: function (width) {return width * 2;},
            hoverWidth: function (width) {return width * 1.5;}
        },
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based', // Good clustering behavior
            forceAtlas2Based: {
                gravitationalConstant: -100, // Repulsion
                centralGravity: 0.015,
                springLength: 200,
                springConstant: 0.08,
                damping: 0.4,
                avoidOverlap: 0.5
            },
            stabilization: {
                enabled: true,
                iterations: 200, // Calculate before rendering
                updateInterval: 50,
                onlyDynamicEdges: false,
                fit: true
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            hideEdgesOnDrag: true, // Improves performance when dragging
            hideEdgesOnZoom: false,
            zoomSpeed: 0.4
        },
        groups: {
            // Define colors for out_degree ranges (dynamically assigned groups 0 to N)
            // Color map transitioning Yellow-Green-Blueish based on original script request (mapped via CSS instead here for JS)
        }
    };

    // Color mapper based on value inside `groups`
    function getColorByOutDegree(degree) {
        // Gradient approx matching YlGnBu context (light yellow to dark blueish/purple)
        if (degree === 0) return { background: '#f6e58d', border: '#eebb22' }; // Yellow
        if (degree < 3) return { background: '#badc58', border: '#6ab04c' }; // Green
        if (degree < 6) return { background: '#7ed6df', border: '#22a6b3' }; // Cyan
        if (degree < 10) return { background: '#686de0', border: '#4834d4' }; // Deep Blue
        return { background: '#a29bfe', border: '#6c5ce7' }; // Purple (Very High)
    }

    // Initialize Network
    function initNetwork() {
        if (network) network.destroy(); // Clear existing
        
        const data = { nodes: nodesDataset, edges: edgesDataset };
        network = new vis.Network(container, data, options);
        
        // --- Event Listeners on Network --- //
        
        // Hide loading screen when physics finishes its initial computation
        network.on("stabilizationIterationsDone", function () {
            loadingOverlay.classList.remove('active');
            statusBadge.textContent = 'Stable';
            statusBadge.classList.add('success');
            network.setOptions( { physics: { enabled: physicsEnabled } } );
        });

        // Track when physics is running
        network.on("startStabilizing", function () {
            statusBadge.textContent = 'Computing...';
            statusBadge.classList.remove('success');
        });
        
    }

    // Load Data from Flask Backend
    function updateDatasetOptions(batchName) {
        const files = datasetConfig.batch_files[batchName] || [];
        datasetSelect.innerHTML = '';
        files.forEach((file) => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            datasetSelect.appendChild(option);
        });
        return files;
    }

    async function loadData(filename, batchName) {
        try {
            loadingOverlay.classList.add('active');
            statusBadge.textContent = 'Fetching...';
            statusBadge.className = 'status-badge';

            const params = new URLSearchParams({ file: filename });
            if (batchName) {
                params.set('batch', batchName);
            }
            const response = await fetch(`/api/data?${params.toString()}`);
            if (!response.ok) throw new Error('Data fetch failed');
            
            const data = await response.json();
            
            // Apply Colors based on custom groups dynamic mapping
            data.nodes.forEach(node => {
                let colors = getColorByOutDegree(node.group);
                
                // Override for Altruistic Donors
                if (node.is_altruistic) {
                    colors = { background: '#ffffff', border: '#aaaaaa' };
                    node.font = { color: '#ffffff', size: 16, bold: true };
                }
                
                node.color = {
                    background: colors.background,
                    border: colors.border,
                    highlight: { background: '#fff', border: colors.border },
                    hover: { background: '#fff', border: colors.border }
                };
            });
                
            // Convert string titles to HTML elements to avoid escaping
            data.nodes.forEach(node => {
                if (node.title) {
                    node.title = createTooltipElement(node.title);
                }
            });

            data.edges.forEach(edge => {
                if (edge.title) {
                    edge.title = createTooltipElement(edge.title);
                }
            });
            
            // Clear and Load
            nodesDataset.clear();
            edgesDataset.clear();
            nodesDataset.add(data.nodes);
            edgesDataset.add(data.edges);
            
            // Update Stats UI
            nodeCountSpan.textContent = data.nodes.length;
            edgeCountSpan.textContent = data.edges.length;
            
            // Initialize graph with data loaded
            initNetwork();
            
        } catch (error) {
            console.error("Error loading data:", error);
            statusBadge.textContent = 'Error';
            statusBadge.style.color = 'var(--danger-color)';
            loadingOverlay.classList.remove('active');
            alert("Failed to load graph data. See console.");
        }
    }

    // UI Event Listeners
    batchSelect.addEventListener('change', (e) => {
        const files = updateDatasetOptions(e.target.value);
        if (files.length > 0) {
            datasetSelect.value = files[0];
            loadData(files[0], e.target.value);
        } else {
            nodeCountSpan.textContent = '-';
            edgeCountSpan.textContent = '-';
            nodesDataset.clear();
            edgesDataset.clear();
            if (network) {
                network.destroy();
                network = null;
            }
            statusBadge.textContent = 'No Files';
        }
    });

    datasetSelect.addEventListener('change', (e) => {
        if(e.target.value) {
            loadData(e.target.value, batchSelect.value);
        }
    });

    btnFit.addEventListener('click', () => {
        if (network) {
            network.fit({ animation: { duration: 800, easingFunction: 'easeInOutQuad' } });
        }
    });

    btnPhysics.addEventListener('click', () => {
        if (!network) {
            return;
        }
        physicsEnabled = !physicsEnabled;
        network.setOptions( { physics: { enabled: physicsEnabled } } );
        
        if (physicsEnabled) {
            btnPhysics.style.background = 'var(--accent-color)';
            btnPhysics.title = "Disable Physics";
        } else {
            btnPhysics.style.background = 'var(--bg-secondary)';
            btnPhysics.title = "Enable Physics";
            statusBadge.textContent = 'Static';
        }
    });

    // --- On Mount --- //
    // If a default file is selected in dropdown, load it
    if (batchSelect.value) {
        updateDatasetOptions(batchSelect.value);
    }
    if(datasetSelect.value) {
        loadData(datasetSelect.value, batchSelect.value || datasetConfig.default_batch);
    }
});
