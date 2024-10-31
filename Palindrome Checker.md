
# Palindrome Checker

This repository contains a simple web application for checking if a given text is a palindrome.

## HTML Code

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Palindrome Checker</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Palindrome Checker</h1>
        <div class="input-group">
            <input type="text" id="text-input" placeholder="Enter text to check">
            <button id="check-btn">Check</button>
        </div>
        <div id="result"></div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

## CSS Code

```css
body {
    font-family: Arial, sans-serif;
    max-width: 600px;
    margin: 2rem auto;
    padding: 0 1rem;
    background-color: #f0f2f5;
}

.container {
    background-color: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h1 {
    color: #1a73e8;
    text-align: center;
    margin-bottom: 1.5rem;
}

.input-group {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

#text-input {
    flex: 1;
    padding: 0.5rem;
    font-size: 1rem;
    border: 2px solid #ddd;
    border-radius: 4px;
}

#check-btn {
    padding: 0.5rem 1rem;
    font-size: 1rem;
    background-color: #1a73e8;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
}

#check-btn:hover {
    background-color: #1557b0;
}

#result {
    padding: 1rem;
    border-radius: 4px;
    background-color: #f8f9fa;
    min-height: 1.5rem;
}
```

## JavaScript Code

```javascript
function isPalindrome(str) {
    // Convert to lowercase and remove all non-alphanumeric characters
    const cleanStr = str.toLowerCase().replace(/[\W_]/g, '');
    
    // Compare the string with its reverse
    const reversedStr = cleanStr.split('').reverse().join('');
    return cleanStr === reversedStr;
}

function checkPalindrome() {
    const textInput = document.getElementById('text-input');
    const result = document.getElementById('result');
    const text = textInput.value;

    // Check if input is empty
    if (!text) {
        alert('Please input a value');
        return;
    }

    // Check if it's a palindrome and display result
    const isPal = isPalindrome(text);
    result.textContent = `${text} is ${isPal ? '' : 'not '}a palindrome`;
}

// Add event listeners
document.getElementById('check-btn').addEventListener('click', checkPalindrome);
document.getElementById('text-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        checkPalindrome();
    }
});
```
