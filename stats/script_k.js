document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const sectionDropdown = document.getElementById('section-dropdown');
    const suggestionsList = document.getElementById('suggestions-list');
    const promptForm = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    const chatResults = document.getElementById('chat-results');
    const deleteChatsBtn = document.getElementById('delete-chats-btn');
    const themeTogglerBtn = document.getElementById('theme-toggler-btn');

    // API Base URL - Change this to match your FastAPI server
    const API_BASE_URL = 'http://127.0.0.1:8000';

    // Icon mapping for suggestion items
    const icons = [
        'draw',
        'lightbulb',
        'explore',
        'list',
        'gesture'
    ];

    // Initialize the app
    init();

    // Initialize function
    async function init() {
        await loadSections(); // Load sections dynamically
        setupEventListeners();
    }

    // Section change handler
    function onSectionChange() {
        const selectedSection = sectionDropdown.value;
        if (selectedSection) {
            fetchTables(selectedSection); // Fetch tables for the selected section
            fetchQuestions(selectedSection); // Fetch questions for the selected section
        }
    }

    // Fetch tables for the selected section
    async function fetchTables(selectedSection) {
        try {
            const response = await fetch(`${API_BASE_URL}/get-tables/?selected_section=${selectedSection}`);
            const data = await response.json();

            if (data.tables && data.tables.length > 0) {
                console.log('Tables fetched:', data.tables);
                // You can use this data to populate a tables dropdown or display available tables
            } else {
                console.error('No tables found for the selected section.');
            }
        } catch (error) {
            console.error('Error fetching tables:', error);
            alert('An error occurred while fetching tables.');
        }
    }

    // Fetch questions for the selected section
    async function fetchQuestions(selectedSection) {
        try {
            const response = await fetch(`${API_BASE_URL}/get_questions/?subject=${selectedSection}`);
            const data = await response.json();

            if (data.questions && data.questions.length > 0) {
                // Display suggestions (questions) in the UI
                displaySuggestions(data.questions);
            } else {
                console.error('No questions found for the selected section.');
                displaySuggestions([]); // Clear suggestions if no questions are found
            }
        } catch (error) {
            console.error('Error fetching questions:', error);
            alert('An error occurred while fetching questions.');
        }
    }

    // Load sections dynamically
    async function loadSections() {
        try {
            const response = await fetch(`${API_BASE_URL}/get-sections`); // Fetch sections from the backend
            const data = await response.json();

            if (data.sections && data.sections.length > 0) {
                // Clear existing options
                sectionDropdown.innerHTML = '<option value="" disabled selected>Select a section</option>';

                // Populate dropdown with fetched sections
                data.sections.forEach(section => {
                    const option = document.createElement('option');
                    option.value = section;
                    option.textContent = section;
                    sectionDropdown.appendChild(option);
                });
            } else {
                console.error('No sections found in the response.');
            }
        } catch (error) {
            console.error('Error loading sections:', error);
            alert('An error occurred while loading sections.');
        }
    }

    // Display suggestions in the UI
    function displaySuggestions(questions) {
        // Clear existing suggestions
        suggestionsList.innerHTML = '';

        // Add new suggestions (limit to 5)
        questions.slice(0, 5).forEach((question, index) => {
            const li = document.createElement('li');
            li.className = 'suggestions-item';

            const p = document.createElement('p');
            p.className = 'text';
            p.textContent = question;

            const span = document.createElement('span');
            span.className = 'material-symbols-rounded';
            span.textContent = icons[index % icons.length];

            li.appendChild(p);
            li.appendChild(span);

            // Add click event to fill the input with the suggestion
            li.addEventListener('click', () => {
                promptInput.value = question;
                // Trigger the input event to show the send button
                promptInput.dispatchEvent(new Event('input'));
            });

            suggestionsList.appendChild(li);
        });
    }

    // Set up event listeners
    function setupEventListeners() {
        // Section dropdown change
        sectionDropdown.addEventListener('change', function () {
            onSectionChange(); // Handle section change
        });

        // Form submission
        promptForm.addEventListener('submit', function (e) {
            e.preventDefault();
            if (promptInput.value.trim()) {
                submitQuery(promptInput.value);
            }
        });

        // Delete chats button
        deleteChatsBtn.addEventListener('click', function () {
            chatResults.innerHTML = '';
        });

        // Theme toggler
        themeTogglerBtn.addEventListener('click', toggleTheme);
    }

    // Submit query to the API
    async function submitQuery(event) {
        const section = document.getElementById("section-dropdown").value;
        const promptInput = document.getElementById("prompt-input");
        const query = promptInput.value.trim();
        if (!section) {
            displayErrorMessage('Please select a section first');
            return;
        }
        if (!query) {
            displayErrorMessage('Please enter a query');
            return;
        }

        // Add user message to chat
        addMessageToChat('user', query);

        // Clear input
        promptInput.value = '';

        // Show loading indicator
        showLoadingIndicator();

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('section', section);
            formData.append('user_query', query);
            // Log FormData contents
            for (let [key, value] of formData.entries()) {
                console.log(key, value);
}
            
            // Send request
            // const response = await 
            const response = await fetch("http://127.0.0.1:8000/submit", {
                method: "POST",
                body: formData,
            })

            
            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();
            console.log(data)
            // Process the response based on the data structure
            processResponse(data);

        } catch (error) {
            console.error('Error submitting query:', error);
            removeLoadingIndicator();
            displayErrorMessage('An error occurred while processing your request');
        }
    }

    // Process API response
    // function processResponse(data) {
    //     console.log(data)
    //     if (data.search_results) {
    //         // Handle researcher or intellidoc response
    //         addMessageToChat('ai', data.search_results);
    //     } else if (data.query) {
    //         // Handle SQL query response
    //         const message = `
    //             <div>
    //                 <p><strong>Generated SQL:</strong></p>
    //                 <pre>${data.query}</pre>
    //             </div>
    //         `;
    //         addMessageToChat('ai', message);

    //         // Display tables if available
    //         if (data.tables && data.tables.length > 0) {
    //             data.tables.forEach(tableData => {
    //                 const tableContainer = document.createElement('div');
    //                 tableContainer.className = 'table-container';

    //                 const tableTitle = document.createElement('h3');
    //                 tableTitle.textContent = tableData.table_name;
    //                 tableContainer.appendChild(tableTitle);

    //                 // Add the table HTML
    //                 tableContainer.innerHTML += tableData.table_html;

    //                 chatResults.appendChild(tableContainer);
    //             });
    //         }
    //     } else if (data.message) {
    //         // Handle other types of messages
    //         addMessageToChat('ai', data.message);
    //     } else {
    //         // Fallback for unknown response format
    //         addMessageToChat('ai', 'I received your request, but I\'m not sure how to display the results.');
    //     }
    // }
    function processResponse(data) {
        // Display the user's query
        document.getElementById("user_query_display").innerHTML = `<strong>Query Asked:</strong> ${data.user_query}`;
    
        // Display the SQL query (if available)
        if (data.query) {
            document.getElementById("sql_query_display").innerHTML = `<strong>SQL Query:</strong> ${data.query}`;
        } else {
            document.getElementById("sql_query_display").innerHTML = "";
        }
    
        // Clear previous results
        document.getElementById("tables_container").innerHTML = "";
    
        // Display tables (if available)
        if (data.tables) {
            data.tables.forEach((table) => {
                let tableHtml = `<h2>${table.table_name}</h2>${table.table_html}`;
                document.getElementById("tables_container").innerHTML += tableHtml;
            });
        }
    
        // Display search results (if available)
        if (data.search_results) {
            // Convert markdown to HTML
            const searchResultsHtml = marked(data.search_results, { breaks: true });
            document.getElementById("tables_container").innerHTML += `
                <div>
                    <h2>Search Results:</h2>
                    <br/>
                    ${searchResultsHtml}
                    <div><i class="fa-regular fa-copy" onclick="copyToClipboard()"></i></div>
                </div>`;
        }
    
        // Show the results section
        document.getElementById("query-results").style.display = "block";
    }

    // Add a message to the chat
    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}-message`;

        // Check if content is HTML
        if (content.includes('<') && content.includes('>')) {
            messageDiv.innerHTML = content;
        } else {
            const p = document.createElement('p');
            p.textContent = content;
            messageDiv.appendChild(p);
        }

        chatResults.appendChild(messageDiv);

        // Scroll to the bottom
        window.scrollTo(0, document.body.scrollHeight);
    }

    // Show loading indicator
    function showLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-indicator';
        loadingDiv.id = 'loading-indicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            loadingDiv.appendChild(dot);
        }

        chatResults.appendChild(loadingDiv);
        window.scrollTo(0, document.body.scrollHeight);
    }

    // Remove loading indicator
    function removeLoadingIndicator() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Display error message
    function displayErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chat-message error-message';

        const p = document.createElement('p');
        p.textContent = message;
        errorDiv.appendChild(p);

        chatResults.appendChild(errorDiv);
        window.scrollTo(0, document.body.scrollHeight);
    }

    // Toggle between light and dark theme
    function toggleTheme() {
        const root = document.documentElement;
        const isDarkTheme = getComputedStyle(root).getPropertyValue('--primary-color').trim() === '#101623';

        if (isDarkTheme) {
            // Switch to light theme
            root.style.setProperty('--text-color', '#333333');
            root.style.setProperty('--subheading-color', '#555555');
            root.style.setProperty('--placeholder-color', '#777777');
            root.style.setProperty('--primary-color', '#ffffff');
            root.style.setProperty('--secondary-color', '#f0f0f0');
            root.style.setProperty('--secondary-hover-color', '#e0e0e0');
            root.style.setProperty('--scrollbar-color', '#cccccc');
            themeTogglerBtn.textContent = 'dark_mode';
        } else {
            // Switch to dark theme
            root.style.setProperty('--text-color', '#edf3ff');
            root.style.setProperty('--subheading-color', '#97a7ca');
            root.style.setProperty('--placeholder-color', '#c3cdde');
            root.style.setProperty('--primary-color', '#101623');
            root.style.setProperty('--secondary-color', '#283045');
            root.style.setProperty('--secondary-hover-color', '#333e58');
            root.style.setProperty('--scrollbar-color', '#626a7f');
            themeTogglerBtn.textContent = 'light_mode';
        }
    }
});