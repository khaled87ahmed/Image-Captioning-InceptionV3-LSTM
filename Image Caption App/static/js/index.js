let selectedImageFile = null; // Variable to store the selected image


// Function to handle image selection or drop
function handleImageFile(file) {
    selectedImageFile = file; // Store the selected file in the variable
    document.getElementById('results').innerHTML = '';
    displayImage(file); // Display the image
}



const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-upload');

dropArea.addEventListener('dragover', (event) => {
    event.preventDefault(); // Prevent default behavior
    dropArea.classList.add('hover'); // Add hover class
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('hover'); // Remove hover class
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault(); // Prevent default behavior
    dropArea.classList.remove('hover'); // Remove hover class

    const files = event.dataTransfer.files; // Get dropped files
    if (files.length > 0) {
        fileInput.files = files; // Assign the dropped files to the input
        handleImageFile(files[0]); // Store and display the dropped image
    }
});

// Existing file input change event listener
fileInput.addEventListener('change', function (event) {
    const files = event.target.files;
    if (files.length > 0) {
        handleImageFile(files[0]); // Store and display the selected image
    }
});

// Function to display the image
function displayImage(file) {
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.getElementById('preview-img');
        img.src = e.target.result; // Set the image source to the file content
        document.getElementById('image-preview').style.display = 'block'; // Show the image preview
    };

    if (file) {
        reader.readAsDataURL(file); // Convert the file to a data URL
    }
}



document.getElementById('generate-btn').addEventListener('click', function () {

    document.getElementById('generate-btn').innerHTML = `
    <div class="spinner-border" role="status">
        <span class="sr-only"></span>
    </div>
`;

    document.getElementById('results').innerHTML = '';

    const formData = new FormData(); // Create a FormData object
    const fileInput = document.getElementById('file-upload');


    if (fileInput.files.length > 0) {
        const file = fileInput.files[0]; // Get the selected image
        formData.append('file', file); // Append the image to the formData

        // Send the image to the server using AJAX
        fetch('/generate_caption', {
            method: 'POST',
            body: formData,
        })
            .then(response => {
                document.getElementById('generate-btn').innerHTML = 'Generate Caption';
                return response.json()
            })
            .then(data => {
                document.getElementById('results').innerHTML = `${data.caption}`
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('results').innerHTML = 'Error generating caption.';
            });
    } else {
        document.getElementById('results').innerHTML = 'No image selected. Please choose an image.';
    }



});
