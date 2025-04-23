
function togglePirateMode(enabled) {
    const body = document.body;
    const music = document.getElementById('pirate-music');
    
    if (enabled) {
        body.classList.add('pirate-mode');
        music.play().catch(e => console.error("Erreur de lecture audio:", e));
    } else {
        body.classList.remove('pirate-mode');
        music.pause();
        music.currentTime = 0;
    }
}
    let currentLanguage = 'english';  // Default language

        function translate() {
            document.querySelectorAll('[data-translate]').forEach(element => {
                const key = element.getAttribute('data-translate');
                if (UI_TRANSLATIONS[currentLanguage][key]) {
                    if (element.tagName === 'INPUT' && element.getAttribute('type') === 'text') {
                        element.placeholder = UI_TRANSLATIONS[currentLanguage][key];
                    } else {
                        element.textContent = UI_TRANSLATIONS[currentLanguage][key];
                    }
                }
            });
        }

        // Load settings and translate on page load
        window.onload = function() {
            loadSettings();
            translate();
        };


        function loadSettings() {
            axios.get('/get_settings')
                .then(function (response) {
                    const data = response.data;
                    populateSelect('root-folder', data.root_folders);
                    populateSelect('quality-profile', data.quality_profiles);
                    populateSelect('plex-library', data.plex_libraries);
                    populateSelect('modelSelect', data.model);
                    
                    // Set the current values
                    document.getElementById('root-folder').value = data.current_settings.root_folder;
                    document.getElementById('quality-profile').value = data.current_settings.quality_profile;
                    document.getElementById('plex-library').value = data.current_settings.plex_library;
                    document.getElementById('language').value = data.current_settings.language;
                    document.getElementById('modelSelect').value = data.current_settings.model;
                    
                    changeLanguage(data.current_settings.language);
                })
                .catch(function (error) {
                    console.error('Error loading settings:', error);
                });
        }
        function populateSelect(id, options) {
            const select = document.getElementById(id);
            select.innerHTML = '';  // Clear existing options
            options.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option.value;
                optionElement.textContent = option.label;
                select.appendChild(optionElement);
            });
        }

        function saveSettings() {
    // Récupère tous les champs potentiels du formulaire de settings
    const settings = {
        root_folder: document.getElementById('root-folder').value,
        quality_profile: document.getElementById('quality-profile').value,
        plex_library: document.getElementById('plex-library').value,
        language: document.getElementById('language').value,
        model: document.getElementById('modelSelect').value,
        media_server: document.getElementById('media_server') ? document.getElementById('media_server').value : undefined,
        plex_url: document.getElementById('plex_url') ? document.getElementById('plex_url').value : undefined,
        plex_token: document.getElementById('plex_token') ? document.getElementById('plex_token').value : undefined,
        jellyfin_url: document.getElementById('jellyfin_url') ? document.getElementById('jellyfin_url').value : undefined,
        jellyfin_api_key: document.getElementById('jellyfin_api_key') ? document.getElementById('jellyfin_api_key').value : undefined,
        jellyfin_user: document.getElementById('jellyfin_user') ? document.getElementById('jellyfin_user').value : undefined,
        radarr_url: document.getElementById('radarr_url') ? document.getElementById('radarr_url').value : undefined,
        radarr_api_key: document.getElementById('radarr_api_key') ? document.getElementById('radarr_api_key').value : undefined,
        llm_provider: document.getElementById('llm_provider') ? document.getElementById('llm_provider').value : undefined,
        groq_api_key: document.getElementById('groq_api_key') ? document.getElementById('groq_api_key').value : undefined,
        ollama_url: document.getElementById('ollama_url') ? document.getElementById('ollama_url').value : undefined,
        openai_api_key: document.getElementById('openai_api_key') ? document.getElementById('openai_api_key').value : undefined
    };
    axios.post('/save_settings', settings)
        .then(function (response) {
            changeLanguage(settings.language);
            bootstrap.Modal.getInstance(document.getElementById('settingsModal')).hide();
        })
        .catch(function (error) {
            console.error('Error saving settings:', error);
        });
}



        function displayResults(data) {
            const movieList = document.getElementById('movie-list');
            movieList.innerHTML = '';
            data.movies.forEach(movie => {
                const movieElement = document.createElement('div');
                movieElement.innerHTML = `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="${movie.title} (${movie.year})" data-imdb-id="${movie.imdb_id}" ${movie.in_library ? 'checked' : ''}>
                        <label class="form-check-label">
                            ${movie.title} (${movie.year}) - ${movie.in_library ? UI_TRANSLATIONS[currentLanguage].in_library : UI_TRANSLATIONS[currentLanguage].discovery}
                        </label>
                    </div>
                `;
                movieList.appendChild(movieElement);
            });
            document.getElementById('results-section').style.display = 'block';
            updateSelectAllButton();
            addCreateCollectionButton(); // Ajoute le bouton pour les collections standard
        }

    function updateSelectAllButton() {
        const checkboxes = document.querySelectorAll('#movie-list input[type="checkbox"]');
        const selectAllBtn = document.getElementById('select-all-btn');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        
        const selectAllText = UI_TRANSLATIONS[currentLanguage].select_all;
        const deselectAllText = UI_TRANSLATIONS[currentLanguage].deselect_all;
        
        selectAllBtn.textContent = allChecked ? deselectAllText : selectAllText;
    }
    document.addEventListener('DOMContentLoaded', function() {
        const selectAllBtn = document.getElementById('select-all-btn');
        
        selectAllBtn.addEventListener('click', function() {
            const checkboxes = document.querySelectorAll('#movie-list input[type="checkbox"]');
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            
            checkboxes.forEach(checkbox => {
                checkbox.checked = !allChecked;
            });
            
            updateSelectAllButton();
        });

        // Initialisation du texte du bouton
        updateSelectAllButton();
    });
    function createCollection() {
        const selectedMovies = Array.from(document.querySelectorAll('#movie-list input:checked')).map(input => input.value);
        const collectionName = document.getElementById('theme-input').value.split(/\s+\d+$/)[0].trim();

            if (selectedMovies.length === 0) {
                alert(UI_TRANSLATIONS[currentLanguage].no_movies_selected);
                return;
            }

            // Afficher l'icône de chargement
            const loadingSpinner = document.createElement('div');
            loadingSpinner.className = 'loading-spinner';
            document.querySelector('#results-section button').appendChild(loadingSpinner);

            axios.post('/create_collection', { collection_name: collectionName, selected_movies: selectedMovies })
                .then(function (response) {
                    alert(UI_TRANSLATIONS[currentLanguage].collection_created);
                    updateCollectionsList();
                })
                .catch(function (error) {
                    console.error('Error:', error);
                    alert(UI_TRANSLATIONS[currentLanguage].collection_creation_error);
                })
                .finally(function () {
                    // Supprimer l'icône de chargement
                    loadingSpinner.remove();
                });
        }

        function updateCollectionsList() {
            axios.get('/collections_status')
                .then(function (response) {
                    const collectionsList = document.getElementById('collections-list');
                    collectionsList.innerHTML = '';
                    response.data.forEach(collection => {
                        const collectionElement = document.createElement('div');
                        collectionElement.classList.add('collection-item');
                        
                        let collectionHtml = `
                            <h3>
                                ${collection.is_letterboxd 
                                    ? '<i class="fa-brands fa-letterboxd icone"></i>' 
                                    : '<i class="fa-solid fa-hard-drive icone"></i></i>'}
                                ${collection.name}
                            </h3>
                        `;
        
                        if (collection.is_letterboxd) {
                            collectionHtml += `
                                <p>${UI_TRANSLATIONS[currentLanguage].url}: <a href="${collection.url}" target="_blank">${collection.url}</a></p>
                                <p>${UI_TRANSLATIONS[currentLanguage].movies_count}: ${collection.movies ? collection.movies.length : 0}</p>
                                <p>${UI_TRANSLATIONS[currentLanguage].last_updated}: ${new Date(collection.last_updated).toLocaleString()}</p>
                            `;
                        } else {
                            const statusKey = collection.status ? collection.status.toLowerCase() : '';
                            let statusTranslation = UI_TRANSLATIONS[currentLanguage][statusKey] || collection.status;
                            let statusHtml = statusTranslation;
        
                            if (statusKey === 'in_progress') {
                                statusHtml += ' <div class="loading-spinner"></div>';
                            }
        
                            collectionHtml += `
                                <p>${UI_TRANSLATIONS[currentLanguage].movies_added}: ${collection.added_count}/${collection.total_count}</p>
                                <p>${UI_TRANSLATIONS[currentLanguage].status}: ${statusHtml}</p>
                            `;
                        }
                        collectionHtml += `
                        <button class="btn btn-danger btn-sm mt-2" onclick="deleteCollection('${collection.name}')">
                            ${UI_TRANSLATIONS[currentLanguage].delete_collection}
                        </button>
                        `;
                        collectionElement.innerHTML = collectionHtml;
                        collectionsList.appendChild(collectionElement);
                    });
                })
                .catch(function (error) {
                    console.error('Error:', error);
                });
        }
        

    let currentExampleIndex = 0;
    
    function changeExample() {
        const exampleWrapper = document.getElementById('example-wrapper');
        const exampleText = document.getElementById('example-text');
        const examples = UI_TRANSLATIONS[currentLanguage].examples;
        
        // Sélectionner un index aléatoire
        const randomIndex = Math.floor(Math.random() * examples.length);
        
        exampleWrapper.classList.add('fade-out');
        
        setTimeout(() => {
            // Utiliser l'index aléatoire pour choisir l'exemple
            exampleText.textContent = examples[randomIndex];
            
            exampleWrapper.classList.remove('fade-out');
            exampleWrapper.classList.add('fade-in');
            
            setTimeout(() => {
                exampleWrapper.classList.remove('fade-in');
            }, 500);
        }, 500); 
    }
    

    setInterval(changeExample, 12000);
    document.addEventListener('DOMContentLoaded', function() {
        const searchButton = document.getElementById('search-button');
        if (searchButton) {
            searchButton.addEventListener('click', searchMovies);
        } else {
            console.error('Search button not found');
        }
    });

    async function searchMovies() {
        showLoading();
        const themeInput = document.getElementById('theme-input');
        const inputText = themeInput.value;
        const optionSelect = document.getElementById('search-option');
        
        if (inputText.startsWith('https://letterboxd.com/')) {
            // C'est une URL Letterboxd
            try {
                const response = await fetch('/process_letterboxd_list', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url: inputText })
                });
    
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
    
                const result = await response.json();
                console.log("Letterboxd result:", result); 
                displayLetterboxdResults(result);
            } catch(error) {
                console.error('Error:', error);
                alert(UI_TRANSLATIONS[currentLanguage].letterboxd_error);
            }
        } else {
        const match = inputText.match(/(\d+)\s*$/);
        const count = match ? parseInt(match[1]) : 10;
        const theme = inputText.replace(/\d+\s*$/, '').trim();
        
        if (!themeInput || !optionSelect) {
            console.error('Theme input or option select not found');
            hideLoading();
            return;
        }

        const data = {
            theme: theme,
            count: count, // ou le nombre que vous préférez
            option: optionSelect.value
        };


        try {
            const response = await fetch('/search_movies', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            displayResults(result);
            } catch(error) {
                console.error('Error:', error);
                let errorMessage = 'An unknown error occurred';
                
                if (error.error === 'configuration_error') {
                    errorMessage = 'There are configuration errors:\n';
                    if (error.details.includes('plex_token_missing')) {
                        errorMessage += '- Plex token is missing or invalid\n';
                    }
                    if (error.details.includes('groq_api_key_missing')) {
                        errorMessage += '- Groq API key is missing or invalid\n';
                    }
                    if (error.details.includes('radarr_api_key_missing')) {
                        errorMessage += '- Radarr API key is missing or invalid\n';
                    }
                    errorMessage += 'Please check your settings and update the missing or invalid information.';
                } else {
                    switch(error.error) {
                        case 'rate_limit_exceeded':
                            errorMessage = 'You have exceeded your Groq API rate limit. Please try again later.';
                            break;
                        case 'invalid_api_key':
                            errorMessage = 'Your Groq API key is invalid. Please check your settings.';
                            break;
                        // ... (autres cas d'erreur existants)
                    }
                }
                
                alert(errorMessage);
            }
        }
        hideLoading();
    }

    function showLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const searchButton = document.getElementById('search-button');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
            const loadingText = loadingOverlay.querySelector('.visually-hidden');
            if (loadingText) {
                loadingText.textContent = UI_TRANSLATIONS[currentLanguage].loading;
            }
        }
        if (searchButton) searchButton.disabled = true;
    }

    function hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        const searchButton = document.getElementById('search-button');
        if (loadingOverlay) loadingOverlay.style.display = 'none';
        if (searchButton) searchButton.disabled = false;
    }
    function deleteCollection(collectionName) {
        if (confirm(UI_TRANSLATIONS[currentLanguage].confirm_delete_collection)) {
            axios.post('/delete_collection', { name: collectionName })
                .then(function (response) {
                    alert(UI_TRANSLATIONS[currentLanguage].collection_deleted);
                    updateCollectionsList();
                })
                .catch(function (error) {
                    console.error('Error:', error);
                    if (error.response) {
                        // The request was made and the server responded with a status code
                        // that falls out of the range of 2xx
                        alert(`${UI_TRANSLATIONS[currentLanguage].delete_collection_error}: ${error.response.data.error}`);
                    } else if (error.request) {
                        // The request was made but no response was received
                        alert(UI_TRANSLATIONS[currentLanguage].delete_collection_network_error);
                    } else {
                        // Something happened in setting up the request that triggered an Error
                        alert(UI_TRANSLATIONS[currentLanguage].delete_collection_unknown_error);
                    }
                });
        }
    }
    function displayLetterboxdResults(data) {
        const movieList = document.getElementById('movie-list');
        movieList.innerHTML = `<h2>${data.collection_name}</h2>`;
        
        if (data.movies && data.movies.length > 0) {
            data.movies.forEach(movie => {
                const movieElement = document.createElement('div');
                movieElement.innerHTML = `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="${movie.title}" ${movie.in_plex ? 'checked' : ''}>
                        <label class="form-check-label">
                            ${movie.title} - ${movie.in_plex ? UI_TRANSLATIONS[currentLanguage].in_library : UI_TRANSLATIONS[currentLanguage].discovery}
                        </label>
                    </div>
                `;
                movieList.appendChild(movieElement);
            });
    
            addCreateCollectionButton(true, data.collection_name, data.letterboxd_url);
        } else {
            movieList.innerHTML += '<p>Aucun film trouvé dans cette liste.</p>';
        }
    
        document.getElementById('results-section').style.display = 'block';
        updateSelectAllButton();
    }
    
    function createLetterboxdCollection(collectionName, letterboxdUrl) {
        showLoading();
        const selectedMovies = Array.from(document.querySelectorAll('#movie-list input:checked')).map(input => {
            return {
                title: input.value,
                in_plex: input.dataset.inPlex === 'true'
            };
        });
    
        fetch('/create_letterboxd_collection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                collection_name: collectionName,
                selected_movies: selectedMovies,
                letterboxd_url: letterboxdUrl
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Collection created successfully:", data);
            alert(UI_TRANSLATIONS[currentLanguage].letterboxd_collection_created);
            updateCollectionsList();
        })
        .catch((error) => {
            console.error('Error:', error);
            alert(UI_TRANSLATIONS[currentLanguage].letterboxd_collection_creation_error);
        })
        .finally(() => {
            hideLoading(); 
        });
    }
function addCreateCollectionButton(isLetterboxd = false, collectionName = '', letterboxdUrl = '') {
    const buttonContainer = document.getElementById('create-collection-button-container');
    buttonContainer.innerHTML = ''; 

    const createButton = document.createElement('button');
    createButton.id = 'create-collection-btn';
    createButton.textContent = UI_TRANSLATIONS[currentLanguage].create_collection;
    createButton.classList.add('btn', 'btn-primary', 'mt-3');
    
    if (isLetterboxd) {
        createButton.onclick = () => createLetterboxdCollection(collectionName, letterboxdUrl);
    } else {
        createButton.onclick = createCollection;
    }

    buttonContainer.appendChild(createButton);
}
    // Initialiser le premier exemple au chargement de la page
    document.addEventListener('DOMContentLoaded', () => {
        const exampleText = document.getElementById('example-text');
        exampleText.textContent = UI_TRANSLATIONS[currentLanguage].examples[0];
    });

            // Fonction pour changer la langue (à appeler lorsque l'utilisateur change de langue)
            function changeLanguage(lang) {
                currentLanguage = lang;
                    if (currentLanguage === "pirate"){
                        togglePirateMode(true);
                    }else{
                        togglePirateMode(false);
                    }
                translate(); // Assurez-vous que cette fonction existe et traduit tous les éléments de l'interface
                changeExample();
                 // Mettre à jour l'exemple immédiatement après le changement de langue
            }

        // Call updateCollectionsList every 30 seconds
        setInterval(updateCollectionsList, 30000);
        document.addEventListener('DOMContentLoaded', loadSettings);

        function startCollectionStatusUpdates() {
            updateCollectionsList();
            setInterval(updateCollectionsList, 5000); // Mise à jour toutes les 5 secondes
        }

        // Appelez cette fonction au chargement de la page
    document.addEventListener('DOMContentLoaded', startCollectionStatusUpdates);


