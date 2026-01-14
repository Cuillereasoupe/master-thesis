let map;
let markers = [];
let lakesData = null;
let corrections = null; // Variable pour stocker les corrections

// Function to calculate bloom percentage for a set of observations
function calculateBloomPercentage(rows) {
    const bloomObservations = rows.filter(r => {
        const bloom = r["observation-obs-bloom"];
        return bloom && bloom.toString().trim() !== '' && bloom.toString().toLowerCase() !== 'no' && bloom.toString().toLowerCase() !== 'none';
    });
    
    return rows.length > 0 ? (bloomObservations.length / rows.length) * 100 : 0;
}

// Function to get color based on bloom percentage
function getColorFromBloomPercentage(percentage) {
    if (percentage <= 50) {
        const ratio = percentage / 50;
        const r = Math.round(0 * (1 - ratio) + 0 * ratio);
        const g = Math.round(123 * (1 - ratio) + 255 * ratio);
        const b = Math.round(255 * (1 - ratio) + 0 * ratio);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        const ratio = (percentage - 50) / 50;
        const r = Math.round(0 * (1 - ratio) + 255 * ratio);
        const g = Math.round(255 * (1 - ratio) + 0 * ratio);
        const b = Math.round(0 * (1 - ratio) + 0 * ratio);
        return `rgb(${r}, ${g}, ${b})`;
    }
}

// Function to get border color (darker version)
function getBorderColorFromBloomPercentage(percentage) {
    if (percentage <= 50) {
        const ratio = percentage / 50;
        const r = Math.round(0 * (1 - ratio) + 0 * ratio);
        const g = Math.round(64 * (1 - ratio) + 128 * ratio);
        const b = Math.round(128 * (1 - ratio) + 0 * ratio);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        const ratio = (percentage - 50) / 50;
        const r = Math.round(0 * (1 - ratio) + 128 * ratio);
        const g = Math.round(128 * (1 - ratio) + 0 * ratio);
        const b = Math.round(0 * (1 - ratio) + 0 * ratio);
        return `rgb(${r}, ${g}, ${b})`;
    }
}

// Load corrections from external JSON file
async function loadCorrections() {
    try {
        const response = await fetch('corrections.json');
        if (!response.ok) {
            corrections = { corrections: {}, force_matches: {} };
            return;
        }
        corrections = await response.json();
    } catch {
        corrections = { corrections: {}, force_matches: {} };
    }
}

// Load geoJSON containing european lakes data (from Overpass)
async function loadLakes() {
    const response = await fetch('lakes.geojson');
    lakesData = await response.json();  
}

// Initialize the map
function initMap() {
    map = L.map('map').setView([46.603354, 1.888334], 6);
    
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }).addTo(map);
    
    L.control.scale({
        metric: true,      // show kilometers
        imperial: false,   // hide miles
        position: 'bottomleft'  // or 'bottomright', 'topleft', 'topright'
    }).addTo(map);
}

// Function to update status
function updateStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

// Function to show info panel with legend
function showInfo(pointCount) {    
    const infoEl = document.getElementById('info');
    
    let legendEl = document.getElementById('legend');
    if (!legendEl) {
        legendEl = document.createElement('div');
        legendEl.id = 'legend';
        legendEl.innerHTML = `
            <div style="margin: 10px 0;">
                <h4>Nombre d'observations:</h4>
                <div style="display: flex; align-items: center; margin: 3px 0;">
                    <div style="width: 10px; height: 10px; border-radius: 50%; background-color: #007BFF; border: 2px solid #004080; margin-right: 10px;"></div>
                    <span>1 observation</span>
                </div>
                <div style="display: flex; align-items: center; margin: 3px 0;">
                    <div style="width: 18px; height: 18px; border-radius: 50%; background-color: #007BFF; border: 2px solid #004080; margin-right: 6px;"></div>
                    <span>5 observations</span>
                </div>
                <div style="display: flex; align-items: center; margin: 3px 0;">
                    <div style="width: 30px; height: 30px; border-radius: 50%; background-color: #007BFF; border: 2px solid #004080; margin-right: 0px;"></div>
                    <span>10+ observations</span>
                </div>
                <h4>Pourcentage d'observation d'algues:</h4>
                <div style="display: flex; align-items: center; margin: 6px 0;">
                    <span style="font-size: 12px; color: #333; margin-right: 6px;">0%</span>
                    <div style="margin: 6px 0; width: 200px; height: 16px; background: linear-gradient(to right, rgb(0,123,255), rgb(0,255,0), rgb(255,0,0)); border: 2px solid #000; border-radius: 1px; margin: 0 6px;"></div>
                    <span style="font-size: 12px; color: #333; margin-left: 6px;">100%</span>
                </div>
            </div>
        `;
        infoEl.appendChild(legendEl);
    }
    
    infoEl.style.display = 'block';
}

// Function to format date from the CSV format to a readable format
function formatDate(dateString) {
    if (!dateString) return '';
    const str = dateString.toString().trim();
    
    // Handle JavaScript Date object or date strings like "Sun 13 July 2025..."
    try {
        const date = new Date(dateString);
        if (!isNaN(date.getTime())) {
            const day = date.getDate().toString().padStart(2, '0');
            const month = (date.getMonth() + 1).toString().padStart(2, '0');
            const year = date.getFullYear();
            return `${day}.${month}.${year}`;
        }
    } catch (e) {
        // Fall through to return original
    }
    
    return dateString; // Return original if all parsing fails
}

function createPopupContent(lakeName, rows) {
    let content = '<div style="max-width: 300px;">';
    content += `<h4>${lakeName}</h4>`;
    content += `<p><strong>${rows.length} observation${rows.length > 1 ? 's' : ''}</strong></p>`;
    
    const bloomPercentage = calculateBloomPercentage(rows);
    content += `<p><strong>Pourcentage d'observations positives:</strong> ${bloomPercentage.toFixed(1)}%</p>`;

    // Only show original names if lakeName = "Unknown lake"
    if (lakeName === "Unknown lake") {
        const originalNames = Array.from(new Set(
            rows
                .map(r => r["general-contexte-nom_lac"])
                .filter(v => v && v.toString().trim() !== '')
        ));

        if (originalNames.length > 0) {
            content += "<p><strong>Noms d'origine:</strong></p><ul>";
            originalNames.forEach(n => {
                content += `<li>${n}</li>`;
            });
            content += "</ul>";
        }
    }

    // Create a mapping of images to their corresponding dates
    const imageData = rows
        .map(r => ({
            image: r["photo-photo-img1"],
            date: r["general-contexte-date_heure"]
        }))
        .filter(item => item.image && item.image.toString().trim() !== '')
        .filter((item, index, self) => 
            self.findIndex(i => i.image === item.image) === index // Remove duplicates based on image name
        );

    if (imageData.length > 0) {
        content += "<p><strong>Images:</strong></p>";
        content += '<div style="display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0;">';
        
        imageData.forEach((item, index) => {
            const imagePath = `media/${item.image}`;
            const formattedDate = formatDate(item.date);
            content += `
                <div style="position: relative;">
                    <img src="${imagePath}" 
                         alt="Lake image ${formattedDate || index + 1}" 
                         style="width: 80px; height: 60px; object-fit: cover; border-radius: 4px; cursor: pointer; border: 2px solid #ddd;"
                         onclick="openImageModal('${imagePath}', '${formattedDate || `Image ${index + 1}`}')"
                         onload="this.style.border='2px solid #007BFF'">
                </div>
            `;
        });
        
        content += '</div>';
        
        if (imageData.length > 0) {
            content += `<p style="font-size: 12px; color: #666;">Cliquez sur les images pour agrandir</p>`;
        }
    }

    content += '</div>';
    return content;
}

// Function to open image in modal
function openImageModal(imagePath, caption) {
    let modal = document.getElementById('imageModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'imageModal';
        modal.style.cssText = `
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            cursor: pointer;
        `;
        
        modal.innerHTML = `
            <div style="position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                <img id="modalImage" style="max-width: 90%; max-height: 90%; object-fit: contain;">
                <div id="modalCaption" style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); color: white; background: rgba(0,0,0,0.7); padding: 10px 20px; border-radius: 4px; text-align: center;"></div>
                <span style="position: absolute; top: 15px; right: 35px; color: white; font-size: 40px; font-weight: bold; cursor: pointer;">&times;</span>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        modal.onclick = function() {
            modal.style.display = 'none';
        };
    }
    
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    
    modalImg.src = imagePath;
    modalCaption.textContent = caption;
    modal.style.display = 'block';
}

// Function to apply corrections to a row based on KEY
function applyCorrections(row) {
    const key = row["KEY"];
    if (!key || !corrections?.corrections?.[key]) {
        return row; 
    }
    
    const correction = corrections.corrections[key];
    const correctedRow = { ...row }; 
    
    switch (correction.type) {
        case 'rename_lake':
            correctedRow["general-contexte-nom_lac"] = correction.new_lake_name;
            break;
        case 'force_coordinates':
            if (correction.new_lat && correction.new_lon) {
                correctedRow["general-contexte-loc-Latitude"] = correction.new_lat;
                correctedRow["general-contexte-loc-Longitude"] = correction.new_lon;
            }
            break;
        case 'no_match':
            correctedRow._no_match = true;
            break;
        case 'ignore':
            correctedRow._ignore = true;
            break;
    }
    
    return correctedRow;
}

// Function to load and process CSV from project files
async function loadCSV() {
    try {
        const response = await fetch('obs_algues_bis.csv');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const csvData = await response.text();
        
        Papa.parse(csvData, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            delimitersToGuess: [',', ';', '\t'],
            complete: function(results) {
                processData(results.data);
            }
        });
    } catch (error) {
        updateStatus('Error loading CSV file', 'error');
    }
}

// Function to process the parsed data
function processData(data) {    
    const cleanedData = data.map(row => {
        const cleanedRow = {};
        for (const [key, value] of Object.entries(row)) {
            cleanedRow[key.trim()] = value;
        }
        return cleanedRow;
    });

    markers.forEach(m => map.removeLayer(m));
    markers = [];

    const lakeGroups = {};
    let totalObservations = 0;
    let correctionCount = 0;
    let ignoredCount = 0;
    let noMatchCount = 0;

    cleanedData.forEach((row, index) => {
        const correctedRow = applyCorrections(row);
        if (correctedRow !== row) correctionCount++;
        
        if (correctedRow._ignore) {
            ignoredCount++;
            return;
        }
        
        const lat = parseFloat(String(correctedRow["general-contexte-loc-Latitude"]).replace(',', '.'));
        const lon = parseFloat(String(correctedRow["general-contexte-loc-Longitude"]).replace(',', '.'));
        if (isNaN(lat) || isNaN(lon)) return;

        totalObservations++;

        if (correctedRow._no_match) {
            noMatchCount++;
            const noMatchId = `no_match_${correctedRow.KEY || index}_${lat.toFixed(6)}_${lon.toFixed(6)}`;
            lakeGroups[noMatchId] = {
                lat: lat,
                lon: lon,
                name: correctedRow["general-contexte-nom_lac"] || "Individual point",
                rows: [correctedRow]
            };
            return;
        }

        const originalLakeName = correctedRow["general-contexte-nom_lac"] || "No name in CSV";
        const point = turf.point([lon, lat]);

        const nearestLakes = [];
        lakesData.features.forEach(feature => {
            const centroid = turf.centroid(feature);
            const dist = turf.distance(point, centroid, {units: 'kilometers'});
            nearestLakes.push({ feature, distance: dist });
        });

        nearestLakes.sort((a, b) => a.distance - b.distance);
        const closestFour = nearestLakes.slice(0, 4);

        function namesMatch(original, geoJsonName) {
            if (!original || !geoJsonName) return false;
            
            function extractLakeName(str) {
                const normalized = str.toLowerCase()
                    .replace(/[àáâãäå]/g, 'a')
                    .replace(/[èéêë]/g, 'e')
                    .replace(/[ìíîï]/g, 'i')
                    .replace(/[òóôõö]/g, 'o')
                    .replace(/[ùúûü]/g, 'u')
                    .replace(/[ç]/g, 'c')
                    .replace(/[ñ]/g, 'n')
                    .replace(/[^a-z0-9\s]/g, '')
                    .replace(/\s+/g, ' ')
                    .trim();
                
                const wordsToRemove = [
                    'lac', 'lake', 'loch', 'lough', 'meer', 'see', 'lago', 'laguna',
                    'etang', 'étang', 'pond', 'reservoir', 'reservoire', 'réservoir',
                    'barrage', 'dam', 'retenue', 'bassin', 'basin',
                    'de', 'du', 'des', 'le', 'la', 'les', 'of', 'the', 'von', 'van',
                    'saint', 'st', 'sainte', 'ste', 'san', 'santa', 'sankt'
                ];
                
                const words = normalized.split(' ')
                    .filter(word => word.length > 1)
                    .filter(word => !wordsToRemove.includes(word));
                
                return words.join(' ').trim();
            }
            
            const origCore = extractLakeName(original);
            const geoCore = extractLakeName(geoJsonName);
            
            if (!origCore || !geoCore) return false;
            if (origCore === geoCore) return true;
            if (origCore.includes(geoCore) || geoCore.includes(origCore)) return true;
            
            const origWords = origCore.split(' ').filter(w => w.length > 2);
            const geoWords = geoCore.split(' ').filter(w => w.length > 2);
            
            for (const origWord of origWords) {
                for (const geoWord of geoWords) {
                    if (origWord.length > 3 && geoWord.length > 3) {
                        if (origWord === geoWord || origWord.includes(geoWord) || geoWord.includes(origWord)) {
                            return true;
                        }
                    }
                }
            }
            
            return false;
        }

        function checkForcedMatch(originalName) {
            if (!corrections?.force_matches) return null;
            
            const normalizedOriginal = originalName.toLowerCase()
                .replace(/[àáâãäå]/g, 'a')
                .replace(/[èéêë]/g, 'e')
                .replace(/[ìíîï]/g, 'i')
                .replace(/[òóôõö]/g, 'o')
                .replace(/[ùúûü]/g, 'u')
                .replace(/[ç]/g, 'c')
                .replace(/[ñ]/g, 'n')
                .replace(/[^a-z0-9\s]/g, '')
                .trim();
            
            for (const [key, forcedName] of Object.entries(corrections.force_matches)) {
                if (normalizedOriginal.includes(key)) {
                    return forcedName;
                }
            }
            return null;
        }

        let selectedLake = null;

        const forcedMatch = checkForcedMatch(originalLakeName);
        if (forcedMatch) {
            let bestMatch = null;
            let minDist = Infinity;
            
            lakesData.features.forEach(feature => {
                const lakeName = feature.properties.name
                if (lakeName && namesMatch(forcedMatch, lakeName)) {
                    const centroid = turf.centroid(feature);
                    const dist = turf.distance(point, centroid, { units: 'kilometers' });
                    if (dist < minDist) {
                        minDist = dist;
                        bestMatch = { feature, distance: dist };
                    }
                }
            });

            if (bestMatch) {
                selectedLake = bestMatch;
            }
        }

        // If no forced match, proceed with normal matching
        if (!selectedLake && originalLakeName && originalLakeName.trim() !== '') {
            for (const candidate of closestFour) {
                const candidateName = candidate.feature.properties.name;
                if (namesMatch(originalLakeName, candidateName)) {
                    selectedLake = candidate;
                    break;
                }
            }
        }

        // If no name match found, use the closest lake
        if (!selectedLake && closestFour.length > 0) {
            selectedLake = closestFour[0];
        }

        if (selectedLake) {
            const matchedLakeName = selectedLake.feature.properties.name || "Unknown lake";
            const centroid = turf.centroid(selectedLake.feature).geometry.coordinates;

            const lakeId = selectedLake.feature.properties.id ||
                        `${matchedLakeName}_${centroid[0].toFixed(6)}_${centroid[1].toFixed(6)}`;

            if (!lakeGroups[lakeId]) {
                lakeGroups[lakeId] = {
                    lat: centroid[1],
                    lon: centroid[0],
                    name: matchedLakeName,
                    rows: []
                };
            }
            lakeGroups[lakeId].rows.push(correctedRow);
        }
    });

    // Place one circle per lake (aggregated markers)
    let validLakes = 0;
    for (const { lat, lon, name, rows } of Object.values(lakeGroups)) {
        const size = 5 + rows.length * 2;
        const bloomPercentage = calculateBloomPercentage(rows);
        const fillColor = getColorFromBloomPercentage(bloomPercentage);
        const borderColor = getBorderColorFromBloomPercentage(bloomPercentage);

        const marker = L.circleMarker([lat, lon], {
            radius: size,
            fillColor: fillColor,
            color: borderColor,
            weight: 2,
            opacity: 1,
            fillOpacity: 0.7
        }).addTo(map);

        marker.bindPopup(createPopupContent(name, rows));
        markers.push(marker);
        validLakes++;
    }

    // Calculate mean and SD observations per lake
    const obsCounts = Object.values(lakeGroups).map(g => g.rows.length);
    const meanObs = obsCounts.reduce((a, b) => a + b, 0) / obsCounts.length;
    const variance = obsCounts.reduce((sum, val) => sum + Math.pow(val - meanObs, 2), 0) / obsCounts.length;
    const sdObs = Math.sqrt(variance);

    console.log(`Lakes: ${obsCounts.length}`);
    console.log(`Mean observations per lake: ${meanObs.toFixed(2)} ± ${sdObs.toFixed(2)}`);
    console.log(`Min: ${Math.min(...obsCounts)}, Max: ${Math.max(...obsCounts)}`);

    const lakesWithAlgae = Object.values(lakeGroups).filter(g => 
        g.rows.some(r => r["observation-obs-bloom"] === "yes")
    ).length;
    console.log(`Lakes with ≥1 algae observation: ${lakesWithAlgae} of ${obsCounts.length}`);


    // Analyze repeat observations
    const repeatLakes = Object.values(lakeGroups).filter(g => g.rows.length > 1);
    console.log(`\n=== Repeat observation analysis ===`);
    console.log(`Lakes with multiple observations: ${repeatLakes.length}`);

    repeatLakes.forEach(lake => {
        console.log(`\n${lake.name} (${lake.rows.length} obs):`);
        lake.rows.forEach(r => {
            const date = r["general-contexte-date_heure"] || "no date";
            const bloom = r["observation-obs-bloom"] || "unknown";
            const coverage = r["observation-algae-alg_recouvr_pourcent"] || "-";
            const user = r["version"] || "unknown";
            console.log(`  - ${date} | bloom: ${bloom} | coverage: ${coverage}% | user: ${user}`);
        });
    });

    if (markers.length > 0) {
        const group = new L.featureGroup(markers);
        map.fitBounds(group.getBounds().pad(0.1));

        // Status message in French (viewer info)
        let statusMessage = `${totalObservations} observations chargées sur ${validLakes} lacs.`;
        
        updateStatus(statusMessage, 'success');
        showInfo(validLakes);
    } else {
        updateStatus('Aucune coordonnée valide trouvée', 'error');
    }
}

// Initialize everything when page loads
window.addEventListener('load', async function () {
    initMap();
    await loadCorrections();
    await loadLakes();
    loadCSV();
});