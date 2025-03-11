document.addEventListener('DOMContentLoaded', () => {
    const selectedSectionSelect = document.getElementById('selected-section');
    let selectedSection = selectedSectionSelect.value;

    // Update the collection name based on selected section
    const collectionNameSpan = document.getElementById('collection-name');
    function updateCollectionName() {
        collectionNameSpan.textContent = selectedSection + '_documents';
    }

    // Button and Section Elements
    const showDocsBtn = document.getElementById('show-docs-btn');
    const deleteDocBtn = document.getElementById('delete-doc-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const getTablesBtn = document.getElementById('get-tables-btn');
    const actualUploadBtn = document.getElementById('actual-upload-btn');
    const actualDeleteBtn = document.getElementById('actual-delete-btn');

    const showSection = document.getElementById('show-documents-section');
    const deleteSection = document.getElementById('delete-section');
    const uploadSection = document.getElementById('upload-section');
    const getTablesSection = document.getElementById('get-tables-section');

    const documentsListDiv = document.getElementById('documents-list');
    const deleteResultDiv = document.getElementById('delete-result');
    const uploadResultDiv = document.getElementById('upload-result');
    const tablesListDiv = document.getElementById('tables-list');

    const docToDeleteSelect = document.getElementById('doc-to-delete');

    // Helper Functions
    async function fetchData(url, method = 'GET', body = null, headers = {}) {
        try {
            const response = await fetch(url, {
                method: method,
                headers: headers,
                body: body
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Fetch error:', error);
            throw error;
        }
    }

    async function populateDocuments() {
        try {
            const data = await fetchData('/show_documents', 'POST', `section=${selectedSection}`, {
                'Content-Type': 'application/x-www-form-urlencoded'
            });

            docToDeleteSelect.innerHTML = ''; // Clear previous options
            data.forEach(doc => {
                const option = document.createElement('option');
                option.value = doc;
                option.textContent = doc;
                docToDeleteSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error populating documents:', error);
            docToDeleteSelect.innerHTML = '<option>Error loading documents</option>';
        }
    }

    // Function to hide all content sections
    function hideSections() {
        showSection.classList.add('hidden');
        deleteSection.classList.add('hidden');
        uploadSection.classList.add('hidden');
        getTablesSection.classList.add('hidden');
    }

    // Event Listeners

    // Listen for changes in the section selection
    selectedSectionSelect.addEventListener('change', () => {
        selectedSection = selectedSectionSelect.value;
        updateCollectionName(); // Update collection name
        hideSections(); //Hide all section on change
    });

    // Button Click Event Listeners

    showDocsBtn.addEventListener('click', async () => {
        hideSections();
        showSection.classList.remove('hidden');
        try {
            const data = await fetchData('/show_documents', 'POST', `section=${selectedSection}`, {
                'Content-Type': 'application/x-www-form-urlencoded'
            });
            documentsListDiv.textContent = 'Documents: ' + data.join(', ');
        } catch (error) {
            documentsListDiv.textContent = 'Error fetching documents.';
        }
    });

    uploadBtn.addEventListener('click', () => {
        hideSections();
        uploadSection.classList.remove('hidden');
    });

    actualUploadBtn.addEventListener('click', async () => {
        const files = document.getElementById('file-upload').files;

        if (files.length === 0) {
            uploadResultDiv.textContent = 'Please select files to upload.';
            return;
        }

        const formData = new FormData();
        formData.append('section', selectedSection);
        for (const file of files) {
            formData.append('files', file);
        }

        try {
            const data = await fetchData('/upload', 'POST', formData); // No extra headers needed for FormData
            uploadResultDiv.textContent = data.message;
        } catch (error) {
            uploadResultDiv.textContent = 'Error uploading files.';
        }
    });

    deleteDocBtn.addEventListener('click', () => {
        hideSections();
        deleteSection.classList.remove('hidden');
        populateDocuments();
    });

    actualDeleteBtn.addEventListener('click', async () => {
        const docName = docToDeleteSelect.value;
        try {
            const data = await fetchData('/delete_document', 'POST', `section=${selectedSection}&doc_name=${docName}`, {
                'Content-Type': 'application/x-www-form-urlencoded'
            });
            deleteResultDiv.textContent = data.message;
        } catch (error) {
            deleteResultDiv.textContent = 'Error deleting document.';
        }
    });

    getTablesBtn.addEventListener('click', async () => {
        hideSections();
        getTablesSection.classList.remove('hidden');
        try {
            const data = await fetchData(`/get-tables/?selected_section=${selectedSection}`);
            tablesListDiv.textContent = 'Tables: ' + data.tables.join(', ');
        } catch (error) {
            tablesListDiv.textContent = 'Error fetching tables.';
        }
    });
    // Initialization
    updateCollectionName(); // Call initially to display default section
    hideSections()
    selectedSection = selectedSectionSelect.value;
});
