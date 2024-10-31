
# US Phone Number Validator

This project is a web application designed to validate US phone numbers. Users can input a phone number in various accepted formats, and the application will determine if the number is valid or not.

## HTML Code

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>US Phone Number Validator</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>US Phone Number Validator</h1>
        <div class="validator-box">
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Enter phone number">
                <div class="buttons">
                    <button id="check-btn">Validate</button>
                    <button id="clear-btn">Clear</button>
                </div>
            </div>
            <div id="results-div"></div>
        </div>
        <div class="examples">
            <h2>Valid Formats</h2>
            <ul>
                <li>1 555-555-5555</li>
                <li>1 (555) 555-5555</li>
                <li>5555555555</li>
                <li>555-555-5555</li>
                <li>(555)555-5555</li>
                <li>1(555)555-5555</li>
                <li>1 555 555 5555</li>
            </ul>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

## CSS Code

```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f0f2f5;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.container {
    max-width: 600px;
    width: 100%;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    text-align: center;
    margin-bottom: 30px;
}

.validator-box {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.input-group {
    margin-bottom: 20px;
}

#user-input {
    width: 100%;
    padding: 12px;
    font-size: 16px;
    border: 2px solid #ddd;
    border-radius: 5px;
    margin-bottom: 15px;
    outline: none;
}

#user-input:focus {
    border-color: #3498db;
}

.buttons {
    display: flex;
    gap: 10px;
}

button {
    flex: 1;
    padding: 12px;
    font-size: 16px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.2s;
}

#check-btn {
    background-color: #3498db;
    color: white;
}

#check-btn:hover {
    background-color: #2980b9;
}

#clear-btn {
    background-color: #e74c3c;
    color: white;
}

#clear-btn:hover {
    background-color: #c0392b;
}

#results-div {
    min-height: 40px;
    padding: 15px;
    background-color: #f8f9fa;
    border-radius: 5px;
    font-size: 16px;
    word-break: break-all;
}

.examples {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.examples h2 {
    color: #2c3e50;
    font-size: 1.2em;
    margin-bottom: 15px;
}

.examples ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
}

.examples li {
    background-color: #f8f9fa;
    padding: 8px;
    border-radius: 5px;
    font-family: monospace;
    text-align: center;
}
```

## JavaScript Code

```javascript
function validatePhoneNumber(phoneNumber) {
    // Regex for validating US phone numbers
    const phoneRegex = /^(1\s?)?(\(\d{3}\)|\d{3})[-\s]?(\d{3})[-\s]?(\d{4})$/;
    
    // Remove any non alphanumeric characters for additional checking
    const cleaned = phoneNumber.replace(/\D/g, '');
    
    // Check if the number matches the regex pattern and has correct length
    if (!phoneRegex.test(phoneNumber)) {
        return false;
    }
    
    // Additional validation for country code
    if (cleaned.length === 11 && cleaned[0] !== '1') {
        return false;
    }
    
    if (cleaned.length > 11) {
        return false;
    }
    
    return true;
}

function checkNumber() {
    const userInput = document.getElementById('user-input').value;
    const resultsDiv = document.getElementById('results-div');
    
    // Check if input is empty
    if (!userInput) {
        alert('Please provide a phone number');
        return;
    }
    
    // Validate the phone number
    const isValid = validatePhoneNumber(userInput);
    
    // Display the result
    resultsDiv.textContent = `${isValid ? 'Valid' : 'Invalid'} US number: ${userInput}`;
}

function clearResults() {
    document.getElementById('user-input').value = '';
    document.getElementById('results-div').textContent = '';
}

// Add event listeners
document.getElementById('check-btn').addEventListener('click', checkNumber);
document.getElementById('clear-btn').addEventListener('click', clearResults);
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        checkNumber();
    }
});
```

---

This README includes the full HTML, CSS, and JavaScript code blocks for the US Phone Number Validator project.
