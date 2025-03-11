console.log("hello");

document.addEventListener('DOMContentLoaded', function () {
    // ✅ Get Sidebar Elements
    const sidebar = document.querySelector(".sidebar");
    const sidebarHeader = document.querySelector(".sidebar-header");

    // ✅ Get Other DOM Elements
    const urlParams = new URLSearchParams(window.location.search);
    const userName = urlParams.get("name");
    const userSection = urlParams.get("section");

    const suggestionsList = document.getElementById("suggestions-list");
    const promptForm = document.getElementById("prompt-form");
    const promptInput = document.getElementById("prompt-input");
    const chatResults = document.getElementById("chat-results");
    const deleteChatsBtn = document.getElementById("delete-chats-btn");
    const themeTogglerBtn = document.getElementById("theme-toggler-btn");

    const API_BASE_URL = "http://127.0.0.1:8000";

    const icons = [
        'draw', 'lightbulb', 'explore', 'list', 'gesture',
        'rocket_launch', 'psychology', 'auto_awesome', 'hub', 'insights'
    ];

    // ✅ Sidebar Toggle Functionality (Inside DOMContentLoaded)
    if (sidebar && sidebarHeader) {
        sidebarHeader.addEventListener("click", function () {
            sidebar.classList.toggle("collapsed");
            console.log("Sidebar toggled:", sidebar.classList.contains("collapsed"));
        });
    } else {
        console.error("Sidebar or Sidebar Header not found.");
    }


    // Show user's name
    if (userName) {
        document.getElementById("greeting").textContent = `Hello, ${userName}`;
    }

    // Ensure a section is selected, else show an alert
    if (!userSection) {
        alert("No section selected! Please log in again.");
    } else {
        fetchQuestions(userSection); // Fetch questions based on section
    }

    // Fetch questions for the selected section
    async function fetchQuestions(selectedSection) {
        try {
            const response = await fetch(`${API_BASE_URL}/get_questions/?subject=${selectedSection}`);
            const data = await response.json();

            if (data.questions && data.questions.length > 0) {
                displaySuggestions(data.questions);
            } else {
                console.error("No questions found for the selected section.");
                displaySuggestions([]); // Clear suggestions if no questions found
            }
        } catch (error) {
            console.error("Error fetching questions:", error);
            alert("An error occurred while fetching questions.");
        }
    }

    // Display suggestions in the UI
    function displaySuggestions(questions) {
    suggestionsList.innerHTML = ""; // Clear existing suggestions

    const colors = ["#1d7efd", "#28a745", "#ffc107", "#6f42c1", "#e91e63", 
                    "#28a745", "#ffc107", "#6f42c1", "#e91e63", "red"]; 

    questions.slice(0, 10).forEach((question, index) => {
        const li = document.createElement("li");
        li.className = "suggestions-item";

        const p = document.createElement("p");
        p.className = "text";
        p.textContent = question;

        const span = document.createElement("span");
        span.className = "material-symbols-rounded";
        span.textContent = icons[index % icons.length]; // Set icon
        span.style.color = colors[index]; // Explicitly apply color

        li.appendChild(p);
        li.appendChild(span);

        li.addEventListener("click", () => {
            promptInput.value = question;
            promptInput.dispatchEvent(new Event("input"));
        });

        suggestionsList.appendChild(li);
    });
}


    // Handle form submission
    promptForm.addEventListener("submit", function (e) {
        e.preventDefault(); // Prevent form from reloading the page
        if (promptInput.value.trim()) {
            submitQuery(promptInput.value);
        }
    });

    // Submit query to the API
    async function submitQuery(query) {
        if (!userSection) {
            displayErrorMessage("Please select a section first");
            return;
        }

        // Add user message to chat
        addMessageToChat("user", query);

        // Clear input field
        promptInput.value = "";

        // Show loading indicator
        showLoadingIndicator();

        try {
            const formData = new FormData();
            formData.append("section", userSection); // Section from URL
            formData.append("user_query", query);
            formData.append("example_question", ""); // If required by the backend

            const response = await fetch(`${API_BASE_URL}/submit`, {
                method: "POST",
                body: formData,
                headers: { "Accept": "application/json" }
            });

            removeLoadingIndicator();

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            console.log("Response received:", response);

            const data = await response.json();
            console.log("Response received:", data);

            processResponse(data);
        } catch (error) {
            console.error("Error submitting query:", error);
            removeLoadingIndicator();
            displayErrorMessage("An error occurred while processing your request");
        }
    }

    // Process API response
    // function processResponse(data) {
    //     console.log("Processing response:", data);
    
    //     if (data.search_results) {
    //         addMessageToChat("ai", data.search_results);
    //     } else if (data.query) {
    //         const message = `
    //             <div>
    //                 <p><strong>Generated SQL:</strong></p>
    //                 <pre>${data.query}</pre>
    //             </div>
    //         `;
    //         addMessageToChat("ai", message);
    //     } 
        
    //     // If the table_html key exists, display the table
    //     if (data.tables && data.tables.length > 0 && data.tables[0].table_html) {
    //         console.log("Received Table HTML:", data.tables[0].table_html);
    //         addTableToChat(data.tables[0].table_html);  // Call function to add table
    //     } else if (data.message) {
    //         addMessageToChat("ai", data.message);
    //     } else {
    //         addMessageToChat("ai", "I received your request, but I'm not sure how to display the results.");
    //     }
    // }
    
    function processResponse(data) {
        console.log("Processing response:", data);
    
        if (data.search_results) {
            addMessageToChat("ai", data.search_results);
        } 
        if (data.query) {
            const message = `
                <div>
                    <p><strong>Generated SQL:</strong></p>
                    <pre>${data.query}</pre>
                </div>
            `;
            addMessageToChat("ai", message);
        }
    
        if (data.tables && data.tables.length > 0) {
            console.log("Received Table HTML:", data.tables[0].table_html);
            addTableToChat(data.tables[0].table_html);  // Pass the entire table object
        } 
        // else if (data.message) {
        //     addMessageToChat("ai", data.message);
        // } 
        
    }
    
    // function addTableToChat(tableHTML) {
    //     const tableDiv = document.createElement("div");
    //     tableDiv.className = "chat-message ai-message table-container";
    //     tableDiv.innerHTML = tableHTML;  // Inject as raw HTML
    
    //     chatResults.appendChild(tableDiv);
    
    //     // Ensure the page scrolls to the bottom after adding the table
    //     window.scrollTo(0, document.body.scrollHeight);
    // }
    // function addTableToChat(tableHTML) {
    //     const parser = new DOMParser();
    //     const doc = parser.parseFromString(tableHTML, "text/html");
    //     const table = doc.querySelector("table");
    
    //     if (!table) {
    //         console.error("Table not found in the provided HTML.");
    //         return;
    //     }
    
    //     const rows = Array.from(table.querySelectorAll("tbody tr"));
    //     const rowsPerPage = 5;
    //     let currentPage = 1;
    
    //     // Create a wrapper div for the table and pagination
    //     const tableDiv = document.createElement("div");
    //     tableDiv.className = "chat-message ai-message table-container";
    
    //     const wrapper = document.createElement("div");
    //     wrapper.className = "table-wrapper";
        
    //     // Table container
    //     const tableContainer = document.createElement("div");
    //     tableContainer.appendChild(table);
        
    //     // Create pagination buttons
    //     const pagination = document.createElement("div");
    //     pagination.className = "pagination-controls";
        
    //     const prevButton = document.createElement("button");
    //     prevButton.innerHTML = "&#8592;"; // Left arrow
    //     prevButton.disabled = true;
    //     prevButton.addEventListener("click", () => changePage(-1));
    
    //     const nextButton = document.createElement("button");
    //     nextButton.innerHTML = "&#8594;"; // Right arrow
    //     nextButton.addEventListener("click", () => changePage(1));
    
    //     pagination.appendChild(prevButton);
    //     pagination.appendChild(nextButton);
    
    //     wrapper.appendChild(tableContainer);
    //     wrapper.appendChild(pagination);
    //     tableDiv.appendChild(wrapper);
    //     chatResults.appendChild(tableDiv);
    
    //     function renderTable() {
    //         const start = (currentPage - 1) * rowsPerPage;
    //         const end = start + rowsPerPage;
    //         const tbody = table.querySelector("tbody");
    //         tbody.innerHTML = ""; // Clear current rows
    
    //         rows.slice(start, end).forEach(row => tbody.appendChild(row));
    
    //         // Enable/disable buttons based on page
    //         prevButton.disabled = currentPage === 1;
    //         nextButton.disabled = end >= rows.length;
    //     }
    
    //     function changePage(step) {
    //         currentPage += step;
    //         renderTable();
    //     }
    
    //     renderTable(); // Initial render
    //     window.scrollTo(0, document.body.scrollHeight);
    // }
    
    function addTableToChat(tableHTML) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(tableHTML, "text/html");
        const table = doc.querySelector("table");
    
        if (!table) {
            console.error("Table not found in the provided HTML.");
            return;
        }
    
        // Remove the extra index column from the header
        const firstHeader = table.querySelector("thead tr th:first-child");
        if (firstHeader && firstHeader.classList.contains("blank")) {
            firstHeader.remove();
        }
    
        // Remove the extra index column from the body
        table.querySelectorAll("tbody tr").forEach(row => {
            const firstCell = row.querySelector("th");
            if (firstCell) {
                firstCell.remove();
            }
        });
    
        const rows = Array.from(table.querySelectorAll("tbody tr"));
        const rowsPerPage = 5;
        let currentPage = 1;
    
        // Create a wrapper div for the table and pagination
        const tableDiv = document.createElement("div");
        tableDiv.className = "chat-message ai-message table-container";
    
        const wrapper = document.createElement("div");
        wrapper.className = "table-wrapper";
    
        // Table container
        const tableContainer = document.createElement("div");
        tableContainer.appendChild(table);
    
        // Create pagination buttons
        const pagination = document.createElement("div");
        pagination.className = "pagination-controls";
    
        const prevButton = document.createElement("button");
        prevButton.innerHTML = "&#8592;"; // Left arrow
        prevButton.disabled = true;
        prevButton.addEventListener("click", () => changePage(-1));
    
        const nextButton = document.createElement("button");
        nextButton.innerHTML = "&#8594;"; // Right arrow
        nextButton.addEventListener("click", () => changePage(1));
    
        pagination.appendChild(prevButton);
        pagination.appendChild(nextButton);
    
        wrapper.appendChild(tableContainer);
        wrapper.appendChild(pagination);
        tableDiv.appendChild(wrapper);
        chatResults.appendChild(tableDiv);
    
        function renderTable() {
            const start = (currentPage - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            const tbody = table.querySelector("tbody");
            tbody.innerHTML = ""; // Clear current rows
    
            rows.slice(start, end).forEach(row => tbody.appendChild(row));
    
            // Enable/disable buttons based on page
            prevButton.disabled = currentPage === 1;
            nextButton.disabled = end >= rows.length;
        }
    
        function changePage(step) {
            currentPage += step;
            renderTable();
        }
    
        renderTable(); // Initial render
        window.scrollTo(0, document.body.scrollHeight);
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
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "loading-indicator";
        loadingDiv.id = "loading-indicator";

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement("span");
            loadingDiv.appendChild(dot);
        }

        chatResults.appendChild(loadingDiv);
        window.scrollTo(0, document.body.scrollHeight);
    }

    // Remove loading indicator
    function removeLoadingIndicator() {
        const loadingIndicator = document.getElementById("loading-indicator");
        if (loadingIndicator) {
            loadingIndicator.remove();
        }
    }

    // Display error message
    function displayErrorMessage(message) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "chat-message error-message";

        const p = document.createElement("p");
        p.textContent = message;
        errorDiv.appendChild(p);

        chatResults.appendChild(errorDiv);
        window.scrollTo(0, document.body.scrollHeight);
    }

    // Toggle theme
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
   
    themeTogglerBtn.addEventListener('click', toggleTheme);

    // Delete chats button functionality
    deleteChatsBtn.addEventListener("click", function () {
        chatResults.innerHTML = "";
    });
});
